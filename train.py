import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC
from torchmetrics import MeanAbsoluteError, MeanSquaredError, R2Score

from models.model import ChatStyleScorer
from datasets.loader import get_dataloader

def train_one_epoch(model, loader, criterion, optimizer, device, lambda_grl=0.3):
    model.train()
    total_loss = 0.0
    # acc_metric = BinaryAccuracy().to(device)
    # f1_metric = BinaryF1Score().to(device)

    mae_metric = MeanAbsoluteError().to(device)
    mse_metric = MeanSquaredError().to(device)
    r2_metric = R2Score().to(device)
    
    null_loss = nn.GaussianNLLLoss(reduction="mean")

    for i, batch in enumerate(tqdm(loader, desc="Train")):
        wave = batch["waveforms"].to(device)
        spk_emb = batch["spk_embeddings"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attention_mask"].to(device)

        # logits = model(wave, spk_emb, mask)
        # loss = criterion(logits, labels)
        
        # score, grl_feat, spk_feat = model(wave, spk_emb, mask)
        # # 主任务 loss（0/1打分）
        # loss_main = criterion(score, labels)
        
        score, logvar, grl_feat, spk_feat = model(wave, spk_emb, mask)
        var = torch.exp(logvar).clamp_min(1e-6)
        loss_main = null_loss(score, labels, var=var)



        # GRL 对抗 sim loss
        sim = F.cosine_similarity(grl_feat, spk_feat, dim=-1)  # shape: (B,)
        loss_grl = torch.mean(sim)

        # 总 loss
        loss = loss_main + lambda_grl * loss_grl
        
        if i % 50 == 0:
            print(f"Consine sim of batch {i} :{sim.mean().item():.4f}")
        


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * wave.size(0)
        # acc_metric.update(torch.sigmoid(score), labels.int())
        # f1_metric.update(torch.sigmoid(score), labels.int())
        mae_metric.update(score, labels)
        mse_metric.update(score, labels)
        r2_metric.update(score, labels)

    avg_loss = total_loss / len(loader.dataset)
    # acc = acc_metric.compute().item()
    # f1 = f1_metric.compute().item()
    mae = mae_metric.compute().item()
    mse = mse_metric.compute().item()
    r2 = r2_metric.compute().item()
    return avg_loss, mae, mse, r2, loss_main, loss_grl


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    mae_metric = MeanAbsoluteError().to(device)
    mse_metric = MeanSquaredError().to(device)
    r2_metric  = R2Score().to(device)

    for batch in tqdm(loader, desc="Val  "):
        wave  = batch["waveforms"].to(device)
        spk_emb = batch["spk_embeddings"].to(device)
        labels  = batch["labels"].to(device).float()
        mask    = batch["attention_mask"].to(device)

        # 模型输出：mu, logvar
        mu, logvar, _, _ = model(wave, spk_emb, mask)

        # 数值保护：exp 之后再 clamp 一次
        var = torch.exp(logvar).clamp_min(1e-6)

        # 高斯NLL：用 mu 和方差
        loss = criterion(mu, labels, var=var)

        # 累计 total_loss（按样本数加权，最后再除以总样本数）
        total_loss += loss.item() * wave.size(0)

        # 评估指标：直接用 mu（回归）
        mae_metric.update(mu, labels)
        mse_metric.update(mu, labels)
        r2_metric.update(mu, labels)

    avg_loss = total_loss / len(loader.dataset)
    mae = mae_metric.compute().item()
    mse = mse_metric.compute().item()
    r2  = r2_metric.compute().item()
    return avg_loss, mae, mse, r2



def save_checkpoint(model, optimizer, epoch, cfg, name="latest",scheduler=None):
    save_dir = cfg.train.save_dir
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.pt")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,

    }, path)


def main(cfg_path: str):
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    model = ChatStyleScorer(cfg).to(device)
    train_loader = get_dataloader(cfg, split="train")
    val_loader = get_dataloader(cfg, split="val")

    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = nn.GaussianNLLLoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    # scheduler 初始化
    sched_cfg = cfg.train.get("scheduler", {})
    sched_name = sched_cfg.get("name", "none")
    scheduler = None

    if sched_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 5),
            gamma=sched_cfg.get("gamma", 0.5),
        )
    elif sched_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=sched_cfg.get("factor", 0.5),
            patience=sched_cfg.get("patience", 3),
            min_lr=sched_cfg.get("min_lr", 1e-6),
            verbose=True
        )
    elif sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.max_epochs,
            eta_min=sched_cfg.get("min_lr", 1e-6)
        )
    elif sched_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler name: {sched_name}")
    
    
    start_epoch = 0
    best_r2 = -1e9

    #  断点恢复
    if cfg.train.get("resume_path"):
        ckpt = torch.load(cfg.train.resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        if "scheduler" in ckpt and ckpt["scheduler"] and scheduler:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"🔁 Resumed from checkpoint at epoch {start_epoch}")

    #  TensorBoard 日志
    writer = SummaryWriter(log_dir=cfg.train.log_dir)

    for epoch in range(start_epoch, cfg.train.max_epochs):
        print(f"\n🔁 Epoch {epoch}")
        train_loss, train_mae, train_mse, train_r2,loss_main, loss_grl = train_one_epoch(model, train_loader, criterion, optimizer, device, lambda_grl=1.0)
        
        val_loss, val_mae, val_mse, val_r2 = validate(model, val_loader, criterion, device)

        # print(f" Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        # print(f" Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        print(f" Train   Loss: {train_loss:.4f}, MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, R2: {train_r2:.4f}")
        print(f" Val   Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, R2: {val_r2:.4f}")

        writer.add_scalar("Loss/train_loss", train_loss, epoch)
        writer.add_scalar("Loss/train_main", loss_main.item(), epoch)
        writer.add_scalar("Loss/train_grl", loss_grl.item(), epoch)
        
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("MAE/val", val_mae, epoch)
        writer.add_scalar("MSE/val", val_mse, epoch)
        writer.add_scalar("R2/val", val_r2, epoch)


        #  保存模型
        save_checkpoint(model, optimizer, epoch, cfg, name=f"epoch_{epoch:03d}", scheduler=scheduler)
        if val_r2 > best_r2:
            best_r2 = val_r2
            save_checkpoint(model, optimizer, epoch, cfg, name="best", scheduler=scheduler)
            print(f" Saved new best checkpoint - epoch : {epoch:03d}")
            
        #更新scheduler
        if scheduler:
            if sched_name == "plateau":
                scheduler.step(val_r2)  # 使用验证集 F1 作为指标
            else:
                scheduler.step()
                
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

    writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to config.yaml")
    args = parser.parse_args()

    main(args.config)