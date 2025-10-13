import os
import json
import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.model import ChatStyleScorer
from datasets.collate import collate_style_batch

# 三个阈值（组/音频两个维度都使用）
THRESHOLDS = [0.67, 0.68, 0.69]


class InferenceDataset(Dataset):
    def __init__(self, data, target_sr=16000):
        self.data = data
        self.target_sr = target_sr

    def __getitem__(self, idx):
        sample = self.data[idx]
        wav_path = sample["wav_path"]
        utt_id = sample["utt_id"]

        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform[0]
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)

        return {"waveform": waveform, "utt_id": utt_id}

    def __len__(self):
        return len(self.data)


def infer_one_dir(model, cfg, device, subdir: Path, output_name: str):
    """对一个目录推理，输出 jsonl：第一行avg，后续倒序样本。"""
    wav_files = sorted(subdir.glob("*.wav"))
    if not wav_files:
        return f"⚠️ 跳过 {subdir}：无音频", None

    data = [{"wav_path": str(wav), "utt_id": wav.stem} for wav in wav_files]
    dataset = InferenceDataset(data, cfg.dataset.target_sr)
    loader = DataLoader(dataset, batch_size=cfg.eval.batch_size, collate_fn=collate_style_batch)

    rows, scores = [], []
    with torch.no_grad():
        for batch in loader:
            wave = batch["waveforms"].to(device)
            mask = batch["attention_mask"].to(device)
            utt_ids = batch["utt_ids"]

            logits = model(wave, attention_mask=mask, inference_only=True)
            probs = torch.sigmoid(logits).cpu().tolist()

            for utt_id, score in zip(utt_ids, probs):
                s = float(round(score, 5))
                rows.append({"utt_id": utt_id, "score": s})
                scores.append(s)

    # 排序
    rows.sort(key=lambda x: x["score"], reverse=True)
    avg = round(sum(scores) / len(scores), 5) if scores else 0.0

    # 写文件
    out_path = subdir / output_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"avg": avg}, ensure_ascii=False) + "\n")
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return f"✅ {subdir.name} 完成：avg={avg}，保存到：{out_path}", {
        "dir": subdir.name,
        "count": len(scores),
        "avg": avg,
        "output_jsonl": str(out_path)
    }


def run_inference_multithread(cfg_path, ckpt_path, root_dir, output_name="chatScore.jsonl", max_workers=4):
    cfg = OmegaConf.load(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = ChatStyleScorer(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"\n🧵 使用多线程并发推理，最大线程数：{max_workers}")

    root_path = Path(root_dir)
    subdirs = [d for d in root_path.iterdir() if d.is_dir()]

    # 单目录模式
    single_dir_mode = False
    if not subdirs:
        wav_files = list(root_path.glob("*.wav"))
        if wav_files:
            subdirs = [root_path]
            single_dir_mode = True
        else:
            print(f"⚠️ 跳过 {root_path}：无音频文件")
            return

    metrics_list = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(infer_one_dir, model, cfg, device, subdir, output_name): subdir.name
            for subdir in subdirs
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="📊 子目录推理中"):
            subdir_name = futures[future]
            try:
                msg, metrics = future.result()
                print(msg)
                if metrics:
                    metrics_list.append(metrics)
            except Exception as e:
                print(f"❌ 子目录 {subdir_name} 出错：{e}")

    # 只有多目录时才写 global.json
    if metrics_list and not single_dir_mode:
        results_dir = root_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        total_count = sum(m["count"] for m in metrics_list)
        weighted_avg = round(sum(m["avg"] * m["count"] for m in metrics_list) / total_count, 5)

        avgs = [m["avg"] for m in metrics_list]
        avg_of_avgs = round(sum(avgs) / len(avgs), 5) if avgs else None
        max_dir_avg = max(avgs) if avgs else None
        min_dir_avg = min(avgs) if avgs else None
        dirs_ge_07 = sum(1 for a in avgs if a >= 0.7)
        dirs_ge_07_ratio = round(dirs_ge_07 / len(avgs), 5) if avgs else 0.0

        # === 全局读取各目录 chatScore.jsonl，做分布 + 失败统计 ===
        all_scores = []
        # 组失败计数初始化
        failed_group_counts = {"lt_0.67": 0, "lt_0.68": 0, "lt_0.69": 0}
        total_groups = len(metrics_list)
        # 音频失败计数初始化
        failed_audio_counts = {"lt_0.67": 0, "lt_0.68": 0, "lt_0.69": 0}
        total_audios = 0

        for m in metrics_list:
            group_has_lt = {0.67: False, 0.68: False, 0.69: False}
            with open(m["output_jsonl"], "r", encoding="utf-8") as f:
                # 跳过第一行 avg
                first = f.readline()
                for line in f:
                    d = json.loads(line)
                    s = d["score"]
                    all_scores.append(s)
                    total_audios += 1
                    # 音频维度失败计数
                    if s < 0.67:
                        failed_audio_counts["lt_0.67"] += 1
                        group_has_lt[0.67] = True
                    if s < 0.68:
                        failed_audio_counts["lt_0.68"] += 1
                        group_has_lt[0.68] = True
                    if s < 0.69:
                        failed_audio_counts["lt_0.69"] += 1
                        group_has_lt[0.69] = True

            # 组维度：任一低于阈值则该组计数+1
            if group_has_lt[0.67]:
                failed_group_counts["lt_0.67"] += 1
            if group_has_lt[0.68]:
                failed_group_counts["lt_0.68"] += 1
            if group_has_lt[0.69]:
                failed_group_counts["lt_0.69"] += 1

        # 全局分布统计（保持你原有的四个桶）
        dist = {"lt_0.6": 0, "0.6-0.7": 0, "0.7-0.8": 0, "ge_0.8": 0}
        for s in all_scores:
            if s < 0.6:
                dist["lt_0.6"] += 1
            elif s < 0.7:
                dist["0.6-0.7"] += 1
            elif s < 0.8:
                dist["0.7-0.8"] += 1
            else:
                dist["ge_0.8"] += 1
        for k in dist:
            dist[k] = {
                "count": dist[k],
                "ratio": round(dist[k] / len(all_scores), 5) if all_scores else 0.0
            }

        # 失败率
        group_failure_rate = {
            k: round((failed_group_counts[k] / total_groups), 5) if total_groups else 0.0
            for k in failed_group_counts
        }
        audio_failure_rate = {
            k: round((failed_audio_counts[k] / total_audios), 5) if total_audios else 0.0
            for k in failed_audio_counts
        }

        global_info = {
            "root": str(root_path),
            "total_count": total_count,
            "weighted_avg": weighted_avg,
            "avg_of_avgs": avg_of_avgs,
            "max_dir_avg": max_dir_avg,
            "min_dir_avg": min_dir_avg,
            "dirs_ge_0.7_ratio": dirs_ge_07_ratio,
            "score_distribution": dist,

            # === 新增：组维度失败统计 ===
            "total_groups": total_groups,
            "failed_group_counts": failed_group_counts,
            "group_failure_rate": group_failure_rate,

            # === 新增：音频维度失败统计 ===
            "total_audios": total_audios,
            "failed_audio_counts": failed_audio_counts,
            "audio_failure_rate": audio_failure_rate
        }

        summary_path = results_dir / "global.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(global_info, f, ensure_ascii=False, indent=2)

        print(f"📦 全局汇总完成：{summary_path}")
    elif single_dir_mode:
        print("ℹ️ 单目录模式：仅生成该目录的 jsonl，不生成 results/global.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--output-name", type=str, default="chatScore.jsonl")
    parser.add_argument("--max-workers", type=int, default=4, help="最大线程数")
    args = parser.parse_args()

    run_inference_multithread(
        args.config,
        args.ckpt,
        args.root_dir,
        output_name=args.output_name,
        max_workers=args.max_workers,
    )