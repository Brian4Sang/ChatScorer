import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC

from models.model import ChatStyleScorer
from datasets.loader import get_dataloader


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    acc_metric = BinaryAccuracy().to(device)
    f1_metric = BinaryF1Score().to(device)

    for batch in tqdm(loader, desc="Train"):
        wave = batch["waveforms"].to(device)
        spk_emb = batch["spk_embeddings"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attention_mask"].to(device)

        logits = model(wave, spk_emb, mask)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * wave.size(0)
        acc_metric.update(torch.sigmoid(logits), labels.int())
        f1_metric.update(torch.sigmoid(logits), labels.int())

    avg_loss = total_loss / len(loader.dataset)
    acc = acc_metric.compute().item()
    f1 = f1_metric.compute().item()
    return avg_loss, acc, f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    acc_metric = BinaryAccuracy().to(device)
    f1_metric = BinaryF1Score().to(device)
    auc_metric = BinaryAUROC().to(device)

    for batch in tqdm(loader, desc="Val  "):
        wave = batch["waveforms"].to(device)
        spk_emb = batch["spk_embeddings"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["attention_mask"].to(device)

        logits = model(wave, spk_emb, mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * wave.size(0)
        prob = torch.sigmoid(logits)
        acc_metric.update(prob, labels.int())
        f1_metric.update(prob, labels.int())
        auc_metric.update(prob, labels.int())

    avg_loss = total_loss / len(loader.dataset)
    acc = acc_metric.compute().item()
    f1 = f1_metric.compute().item()
    auc = auc_metric.compute().item()
    return avg_loss, acc, f1, auc


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

    criterion = torch.nn.BCEWithLogitsLoss()
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
    best_f1 = 0.0

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
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss, val_acc, val_f1, val_auc = validate(model, val_loader, criterion, device)

        print(f"🎯 Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"🎯 Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("F1/train", train_f1, epoch)
        writer.add_scalar("F1/val", val_f1, epoch)
        writer.add_scalar("AUC/val", val_auc, epoch)

        #  保存模型
        save_checkpoint(model, optimizer, epoch, cfg, name="latest", scheduler=scheduler)
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, optimizer, epoch, cfg, name="best", scheduler=scheduler)
            print(" Saved new best checkpoint.")
            
        #更新scheduler
        if scheduler:
            if sched_name == "plateau":
                scheduler.step(val_f1)  # 使用验证集 F1 作为指标
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