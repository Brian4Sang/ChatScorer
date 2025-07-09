# import os
# import json
# import torch
# import torchaudio
# from tqdm import tqdm
# from omegaconf import OmegaConf
# from torch.utils.data import DataLoader
# from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC

# from models.model import ChatStyleScorer
# from datasets.collate import collate_style_batch


# class InferenceDataset(torch.utils.data.Dataset):
#     def __init__(self, data, spk_embedding_path, target_sr=16000):
#         self.data = data
#         self.spk2embedding = torch.load(spk_embedding_path, map_location="cpu")
#         self.target_sr = target_sr

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         wav_path = sample["wav_path"]
#         utt_id = sample["utt_id"]
#         spk_id = sample["spk_id"]
#         label = float(sample["label"])

#         waveform, sr = torchaudio.load(wav_path)
#         if waveform.shape[0] > 1:
#             waveform = waveform.mean(dim=0)
#         else:
#             waveform = waveform[0]
#         if sr != self.target_sr:
#             waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)

#         spk_emb = self.spk2embedding[spk_id]["embedding"]

#         return {
#             "waveform": waveform,
#             "spk_embedding": spk_emb,
#             "utt_id": utt_id,
#             "label": torch.tensor(label, dtype=torch.float32)
#         }

#     def __len__(self):
#         return len(self.data)


# def run_inference_with_metrics(cfg_path, ckpt_path, output_path):
#     cfg = OmegaConf.load(cfg_path)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # 加载模型
#     model = ChatStyleScorer(cfg).to(device)
#     ckpt = torch.load(ckpt_path, map_location=device)
#     model.load_state_dict(ckpt["model"])
#     model.eval()

#     # 加载 test 数据
#     jsonl_paths = cfg.dataset.jsonl_path.test
#     if isinstance(jsonl_paths, str):
#         jsonl_paths = [jsonl_paths]

#     all_data = []
#     for path in jsonl_paths:
#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 all_data.append(json.loads(line.strip()))
#     print(f"✅ 共加载测试样本：{len(all_data)} 条")

#     dataset = InferenceDataset(all_data, cfg.dataset.spk_embedding_path, cfg.dataset.target_sr)
#     loader = DataLoader(dataset, batch_size=cfg.eval.batch_size, collate_fn=collate_style_batch)

#     # 评估指标
#     acc_metric = BinaryAccuracy().to(device)
#     f1_metric = BinaryF1Score().to(device)
#     auc_metric = BinaryAUROC().to(device)

#     results = []

#     with torch.no_grad():
#         for batch in tqdm(loader, desc="🔍 Inference"):
#             wave = batch["waveforms"].to(device)
#             spk_emb = batch["spk_embeddings"].to(device)
#             mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)
#             utt_ids = batch["utt_ids"]

#             logits = model(wave, spk_emb, mask)
#             probs = torch.sigmoid(logits)

#             acc_metric.update(probs, labels.int())
#             f1_metric.update(probs, labels.int())
#             auc_metric.update(probs, labels.int())

#             probs = probs.cpu().tolist()
#             labels = labels.cpu().tolist()

#             for utt_id, score, label in zip(utt_ids, probs, labels):
#                 results.append({
#                     "utt_id": utt_id,
#                     "score": round(score, 5),
#                     "label": int(label)
#                 })

#     acc = acc_metric.compute().item()
#     f1 = f1_metric.compute().item()
#     auc = auc_metric.compute().item()

#     print(f"\n🎯 Test Metrics:")
#     print(f"  - Accuracy: {acc:.4f}")
#     print(f"  - F1 Score: {f1:.4f}")
#     print(f"  - AUC      : {auc:.4f}")

#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     with open(output_path, "w", encoding="utf-8") as f:
#         for item in results:
#             f.write(json.dumps(item, ensure_ascii=False) + "\n")

#     print(f"✅ 每条打分已保存至：{output_path}")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, required=True)
#     parser.add_argument("--ckpt", type=str, required=True)
#     parser.add_argument("--output", type=str, default="infer_results.jsonl")
#     args = parser.parse_args()

#     run_inference_with_metrics(args.config, args.ckpt, args.output)

import os
import json
import torch
import torchaudio
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC

from models.model import ChatStyleScorer
from datasets.collate import collate_style_batch


class InferenceDataset(Dataset):
    def __init__(self, data, spk_embedding_path, target_sr=16000):
        self.data = data
        self.spk2embedding = torch.load(spk_embedding_path, map_location="cpu")
        self.target_sr = target_sr

    def __getitem__(self, idx):
        sample = self.data[idx]
        wav_path = sample["wav_path"]
        utt_id = sample["utt_id"]
        spk_id = sample["spk_id"]
        label = float(sample["label"])

        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform[0]
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)

        spk_emb = self.spk2embedding[spk_id]["embedding"]

        return {
            "waveform": waveform,
            "spk_embedding": spk_emb,
            "utt_id": utt_id,
            "label": torch.tensor(label, dtype=torch.float32)
        }

    def __len__(self):
        return len(self.data)


def run_inference_with_metrics(cfg_path, ckpt_path, output_path):
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = ChatStyleScorer(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 加载测试数据
    jsonl_paths = cfg.dataset.jsonl_path.test
    if isinstance(jsonl_paths, str):
        jsonl_paths = [jsonl_paths]

    all_data = []
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                all_data.append(json.loads(line.strip()))
    print(f"✅ 共加载测试样本：{len(all_data)} 条")

    dataset = InferenceDataset(all_data, cfg.dataset.spk_embedding_path, cfg.dataset.target_sr)
    loader = DataLoader(dataset, batch_size=cfg.eval.batch_size, collate_fn=collate_style_batch)

    # 通用分类指标
    acc_metric = BinaryAccuracy().to(device)
    f1_metric = BinaryF1Score().to(device)
    auc_metric = BinaryAUROC().to(device)

    results = []
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="🔍 Inference"):
            wave = batch["waveforms"].to(device)
            spk_emb = batch["spk_embeddings"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            utt_ids = batch["utt_ids"]

            logits = model(wave, spk_emb, mask)
            probs = torch.sigmoid(logits)

            acc_metric.update(probs, labels.int())
            f1_metric.update(probs, labels.int())
            auc_metric.update(probs, labels.int())
            
            probs = probs.cpu().tolist()
            labels = labels.cpu().tolist()

            for utt_id, score, label in zip(utt_ids, probs, labels):
                results.append({
                    "utt_id": utt_id,
                    "score": round(score, 5),
                    "label": int(label)
                })
                all_scores.append(score)
                all_labels.append(label)

    # 分类评价指标
    acc = acc_metric.compute().item()
    f1 = f1_metric.compute().item()
    auc = auc_metric.compute().item()

    # 新增统计指标：每类样本平均得分
    scores_label_1 = [s for s, l in zip(all_scores, all_labels) if l == 1]
    scores_label_0 = [s for s, l in zip(all_scores, all_labels) if l == 0]

    f1_score = round(sum(scores_label_1) / len(scores_label_1), 5) if scores_label_1 else 0.0
    f0_score = round(sum(scores_label_0) / len(scores_label_0), 5) if scores_label_0 else 0.0

    # 输出指标
    print(f"\n🎯 Test Metrics:")
    print(f"  - Accuracy : {acc:.4f}")
    print(f"  - F1 Score : {f1:.4f}")
    print(f"  - AUC      : {auc:.4f}")
    print(f"  - F1_Score (label==1 avg): {f1_score:.5f}")
    print(f"  - F0_Score (label==0 avg): {f0_score:.5f}")

    # 保存打分
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ 每条打分结果已保存至：{output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="infer_results.jsonl")
    args = parser.parse_args()

    run_inference_with_metrics(args.config, args.ckpt, args.output)
