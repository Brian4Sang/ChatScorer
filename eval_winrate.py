#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import math
from pathlib import Path
from collections import defaultdict
import argparse

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

from models.model import ChatStyleScorer
from datasets.collate import collate_style_batch   # 复用你现有的 collate

# =========================
# Dataset：只负责把必要字段带出来
# =========================
class JsonlScoreSet(Dataset):
    """
    需要字段：utt_id, wav_path, label, （可选）spk_id, duration
    """
    def __init__(self, jsonl_path, target_sr=16000):
        self.items = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                # 基本字段校验
                assert "utt_id" in d and "wav_path" in d and "label" in d, "jsonl 必须包含 utt_id, wav_path, label"
                d.setdefault("spk_id", "NA")
                self.items.append(d)
        self.target_sr = target_sr

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d = self.items[i]
        # 实际波形读取与重采样在 collate_style_batch 里完成（你现成的逻辑）
        waveform, sr = torchaudio.load(d["wav_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform[0]
        if sr != self.target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)
        return {
            "utt_id": d["utt_id"],
            "waveform": waveform,
            "label": torch.tensor(d["label"], dtype=torch.float32),
            "spk_id": d.get("spk_id", "NA"),
        }

# =========================
# 推理：返回 mu / (mu, var)
# =========================
@torch.no_grad()
def infer_all(model, loader, device, utt_meta):
    """
    返回：
        order_utt_ids: 与预测对齐的 utt_id 列表
        mu:  np.array (N,)
        var: np.array (N,)  若模型不返回 logvar，则用 None
    """
    mu_list, var_list, utts = [], [], []

    for batch in loader:
        wave = batch["waveforms"].to(device)
        mask = batch["attention_mask"].to(device)
        utt_ids = batch["utt_ids"]

        out = model(wave, attention_mask=mask, inference_only=True)
        if isinstance(out, tuple) and len(out) >= 2:
            mu = out[0]
            logvar = out[1]
            var = torch.exp(logvar).clamp_min(1e-8)
            var_list.append(var.cpu().numpy())
        else:
            mu = out
            var_list.append(None)

        mu_list.append(mu.cpu().numpy())
        utts.extend(utt_ids)

    mu = np.concatenate(mu_list, axis=0)
    if var_list[0] is None:
        var = None
    else:
        var = np.concatenate(var_list, axis=0)

    # 校验顺序与 meta 的映射
    assert len(mu) == len(utts)
    return utts, mu, var

# =========================
# Pair 采样与过滤
# =========================
def sample_pairs(utts, meta, mode="within_spk", max_pairs=10000, seed=1234):
    """
    meta[utt_id] = {"label": float, "spk_id": str}
    返回：list[(i,j)] 其中 i,j 是索引（对应 utts 的位置），且 i < j
    """
    rng = random.Random(seed)

    # 按 spk 分桶
    spk2idxs = defaultdict(list)
    for idx, u in enumerate(utts):
        spk = meta[u]["spk_id"]
        spk2idxs[spk].append(idx)

    pairs = []

    if mode == "within_spk":
        for spk, idxs in spk2idxs.items():
            if len(idxs) < 2:
                continue
            # 全组合可能太大，随机采样
            all_pairs = []
            for a in range(len(idxs)):
                for b in range(a+1, len(idxs)):
                    all_pairs.append((idxs[a], idxs[b]))
            rng.shuffle(all_pairs)
            pairs.extend(all_pairs)

    elif mode == "cross_spk":
        # 把 spk 列出来两两组合
        spks = [s for s in spk2idxs.keys() if len(spk2idxs[s]) > 0]
        for i in range(len(spks)):
            for j in range(i+1, len(spks)):
                A, B = spk2idxs[spks[i]], spk2idxs[spks[j]]
                if not A or not B:
                    continue
                # 交叉配对（随机）
                for ai in A:
                    bj = rng.choice(B)
                    pairs.append((ai, bj))

    elif mode == "all_pairs":
        N = len(utts)
        for i in range(N):
            for j in range(i+1, N):
                pairs.append((i, j))
    else:
        raise ValueError(f"Unknown pair mode: {mode}")

    rng.shuffle(pairs)
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    return pairs

def evaluate_winrate(utts, mu, var, meta,
                     pairs,
                     epsilon_label=0.05,
                     tie_tol=1e-6,
                     use_uncertainty=False,
                     z=1.96):
    """
    计算 win-rate（方向一致率）
    过滤：
      - 仅当 |y_a - y_b| >= epsilon_label 才计入
      - tie：|pred_diff| < tie_tol 或 |label_diff| < tie_tol（更严格可只对 pred 做 tie）
      - 若 use_uncertainty=True 且 var 不为 None：
            当 |mu_a - mu_b| <= z * sqrt(var_a + var_b) 时判定为“弃权”
    返回：
      dict: {win_rate, counted, wins, ties, abstains, filtered_small_gap}
    """
    wins = 0
    ties = 0
    abstains = 0
    filtered_small_gap = 0
    counted = 0

    mu = np.asarray(mu, dtype=np.float64)
    var = None if var is None else np.asarray(var, dtype=np.float64)

    for i, j in pairs:
        ui, uj = utts[i], utts[j]
        yi, yj = meta[ui]["label"], meta[uj]["label"]
        mi, mj = mu[i], mu[j]

        # 标签差距过滤
        if abs(yi - yj) < epsilon_label:
            filtered_small_gap += 1
            continue

        # 预测差距是否“tie”
        pred_diff = mi - mj
        if abs(pred_diff) < tie_tol:
            ties += 1
            continue

        # 不确定性弃权
        if use_uncertainty and var is not None:
            si = math.sqrt(max(var[i], 1e-8))
            sj = math.sqrt(max(var[j], 1e-8))
            thr = z * math.sqrt(si * si + sj * sj)
            if abs(pred_diff) <= thr:
                abstains += 1
                continue

        # 方向是否一致
        label_diff = yi - yj
        ok = (label_diff > 0 and pred_diff > 0) or (label_diff < 0 and pred_diff < 0)

        counted += 1
        if ok:
            wins += 1

    win_rate = float(wins / counted) if counted > 0 else float("nan")
    return {
        "win_rate": win_rate,
        "counted_pairs": counted,
        "wins": wins,
        "ties": ties,
        "abstains": abstains,
        "filtered_small_gap": filtered_small_gap
    }

# =========================
# 主流程
# =========================
def main(args):
    # 设备 & 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.config)
    model = ChatStyleScorer(cfg).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 数据
    ds = JsonlScoreSet(args.jsonl)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_style_batch
    )

    # 为了不改你的 collate，这里自己维护 utt_id -> (label, spk)
    # collate 返回的 batch 里我们能拿到 utt_ids，然后用这张表回填 label/spk
    meta = {}
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            meta[d["utt_id"]] = {
                "label": float(d["label"]),
                "spk_id": d.get("spk_id", "NA"),
                "duration": d.get("duration", None)
            }

    # 推理
    utts, mu, var = infer_all(model, loader, device, meta)

    # 配对
    pairs = sample_pairs(
        utts=utts,
        meta=meta,
        mode=args.pair_mode,
        max_pairs=args.max_pairs,
        seed=args.seed
    )

    # 计算整体 win-rate
    res_overall = evaluate_winrate(
        utts=utts, mu=mu, var=var, meta=meta, pairs=pairs,
        epsilon_label=args.epsilon_label,
        tie_tol=args.tie_tol,
        use_uncertainty=args.use_uncertainty,
        z=args.z
    )

    # 分 spk（仅在 within_spk 模式下更有意义）
    by_spk = []
    if args.group_by_spk:
        # 将 pairs 按 spk 归类（同说话人才统计）
        spk2pairs = defaultdict(list)
        for i, j in pairs:
            si = meta[utts[i]]["spk_id"]
            sj = meta[utts[j]]["spk_id"]
            if si == sj:
                spk2pairs[si].append((i, j))
        for spk, ps in spk2pairs.items():
            r = evaluate_winrate(
                utts=utts, mu=mu, var=var, meta=meta, pairs=ps,
                epsilon_label=args.epsilon_label,
                tie_tol=args.tie_tol,
                use_uncertainty=args.use_uncertainty,
                z=args.z
            )
            by_spk.append({"spk_id": spk, **r})
        by_spk.sort(key=lambda x: (-(x["win_rate"] if x["win_rate"]==x["win_rate"] else -1), -x["counted_pairs"]))

    # 输出报告
    report = {
        "settings": {
            "pair_mode": args.pair_mode,
            "epsilon_label": args.epsilon_label,
            "tie_tol": args.tie_tol,
            "use_uncertainty": args.use_uncertainty,
            "z": args.z,
            "max_pairs": args.max_pairs,
            "seed": args.seed
        },
        "overall": res_overall,
        "by_spk": by_spk
    }

    out_path = Path(args.out or "winrate_report.json")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 可选：保存 pair 明细，方便抽查
    if args.save_pairs:
        import csv
        csv_path = out_path.with_suffix(".pairs.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["utt_i","utt_j","label_i","label_j","mu_i","mu_j","var_i","var_j"])
            for i, j in pairs:
                ui, uj = utts[i], utts[j]
                yi, yj = meta[ui]["label"], meta[uj]["label"]
                mi, mj = float(mu[i]), float(mu[j])
                vi = float(var[i]) if var is not None else ""
                vj = float(var[j]) if var is not None else ""
                w.writerow([ui, uj, yi, yj, mi, mj, vi, vj])
        print(f"Saved pairs to: {csv_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jsonl", required=True)

    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)

    ap.add_argument("--pair-mode", type=str, default="within_spk",
                    choices=["within_spk", "cross_spk", "all_pairs"])
    ap.add_argument("--epsilon-label", type=float, default=0.05,
                    help="仅统计标签差距 >= 该阈值的样本对")
    ap.add_argument("--tie-tol", type=float, default=1e-6,
                    help="预测差距绝对值低于该阈值视为平局，不计入分母")
    ap.add_argument("--use-uncertainty", action="store_true",
                    help="若模型输出 logvar，则用 z*sqrt(var_i+var_j) 做弃权判定")
    ap.add_argument("--z", type=float, default=1.96, help="不确定性弃权的 z 分数（例如 1.96≈95% CI）")
    ap.add_argument("--max-pairs", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--group-by-spk", action="store_true")
    ap.add_argument("--save-pairs", action="store_true",
                    help="保存 pair 明细 CSV 便于抽查")

    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    main(args)