import os
import json
import torch
import torchaudio
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

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
    print(f"共加载测试样本：{len(all_data)} 条")

    dataset = InferenceDataset(all_data, cfg.dataset.spk_embedding_path, cfg.dataset.target_sr)
    loader = DataLoader(dataset, batch_size=cfg.eval.batch_size, collate_fn=collate_style_batch)

    results = []
    all_scores = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="🔍 Inference"):
            wave = batch["waveforms"].to(device)
            spk_emb = batch["spk_embeddings"].to(device)
            mask = batch["attention_mask"].to(device)
            utt_ids = batch["utt_ids"]

            logits = model(wave, spk_emb, mask)
            probs = torch.sigmoid(logits).cpu().tolist()

            for utt_id, score in zip(utt_ids, probs):
                score = round(score, 5)
                results.append({
                    "utt_id": utt_id,
                    "score": score
                })
                all_scores.append(score)

    # 汇总指标
    avg_score = round(sum(all_scores) / len(all_scores), 5)
    gt_half = sum(s > 0.5 for s in all_scores)
    lt_half = sum(s < 0.5 for s in all_scores)
    prop_gt_half = round(gt_half / len(all_scores), 4)
    prop_lt_half = round(lt_half / len(all_scores), 4)

    top20 = sorted(results, key=lambda x: x["score"], reverse=True)[:20]
    bottom20 = sorted(results, key=lambda x: x["score"])[:20]

    # 打印结果
    print(f"\n 统计指标:")
    print(f"  - 平均分：{avg_score}")
    print(f"  - 得分 > 0.5 数量：{gt_half}（占比 {prop_gt_half}）")
    print(f"  - 得分 < 0.5 数量：{lt_half}（占比 {prop_lt_half}）")

    print("\n 得分最高 Top 20:")
    for item in top20:
        print(f"  {item['utt_id']} ({item['score']})")

    print("\n 得分最低 Bottom 20:")
    for item in bottom20:
        print(f"{item['utt_id']} ({item['score']})")

    # 保存打分结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n每条打分结果已保存至：{output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="infer_results.jsonl")
    args = parser.parse_args()

    run_inference_with_metrics(args.config, args.ckpt, args.output)