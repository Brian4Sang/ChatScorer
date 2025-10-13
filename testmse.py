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
            var_list.append(var.detach().cpu().numpy())
        else:
            mu = out
            var_list.append(None)

        mu_list.append(mu.detach().cpu().numpy())
        utts.extend(utt_ids)

    mu = np.concatenate(mu_list, axis=0).reshape(-1)
    if var_list[0] is None:
        var = None
    else:
        var = np.concatenate(var_list, axis=0).reshape(-1)

    assert len(mu) == len(utts)
    return utts, mu, var

# =========================
# 原有：Pair 采样与胜率评测（保留以兼容）
# =========================
def sample_pairs(utts, meta, mode="within_spk", max_pairs=10000, seed=1234):
    rng = random.Random(seed)
    spk2idxs = defaultdict(list)
    for idx, u in enumerate(utts):
        spk = meta[u]["spk_id"]
        spk2idxs[spk].append(idx)

    pairs = []
    if mode == "within_spk":
        for spk, idxs in spk2idxs.items():
            if len(idxs) < 2:
                continue
            all_pairs = []
            for a in range(len(idxs)):
                for b in range(a+1, len(idxs)):
                    all_pairs.append((idxs[a], idxs[b]))
            rng.shuffle(all_pairs)
            pairs.extend(all_pairs)
    elif mode == "cross_spk":
        spks = [s for s in spk2idxs.keys() if len(spk2idxs[s]) > 0]
        for i in range(len(spks)):
            for j in range(i+1, len(spks)):
                A, B = spk2idxs[spks[i]], spk2idxs[spks[j]]
                if not A or not B:
                    continue
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

        if abs(yi - yj) < epsilon_label:
            filtered_small_gap += 1
            continue

        pred_diff = mi - mj
        if abs(pred_diff) < tie_tol:
            ties += 1
            continue

        if use_uncertainty and var is not None:
            si = math.sqrt(max(var[i], 1e-8))
            sj = math.sqrt(max(var[j], 1e-8))
            thr = z * math.sqrt(si * si + sj * sj)
            if abs(pred_diff) <= thr:
                abstains += 1
                continue

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
# 新增：逐样本回归评测（MSE/MAE/RMSE/R2/相关系数）
# =========================
def evaluate_regression(utts, mu, meta):
    """
    对所有样本计算回归指标，不做任何分桶或配对。
    返回 dict，含 MSE/MAE/RMSE/R2/Pearson/Spearman。
    """
    y_true = np.array([float(meta[u]["label"]) for u in utts], dtype=np.float64).reshape(-1)
    y_pred = np.asarray(mu, dtype=np.float64).reshape(-1)

    assert y_true.shape == y_pred.shape and y_true.size > 0, "空数据或形状不匹配"

    # 基础误差
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))

    # R^2
    y_bar = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_bar) ** 2))
    ss_res = float(np.sum(diff ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # 相关系数（Pearson/Spearman）
    def safe_corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a*a).sum() * (b*b).sum())
        if denom == 0:
            return float("nan")
        return float((a*b).sum() / denom)

    pearson = safe_corr(y_true, y_pred)

    # Spearman：转秩相关（稳定起见用 argsort 两次法）
    def rankdata(x):
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(x), dtype=np.float64)
        # 处理并列：平均名次
        # 找到并列块
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j+1]] == x[order[i]]:
                j += 1
            if j > i:
                avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
                ranks[order[i:j+1]] = avg
            i = j + 1
        return ranks

    r_true = rankdata(y_true)
    r_pred = rankdata(y_pred)
    spearman = safe_corr(r_true, r_pred)

    return {
        "count": int(y_true.size),
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman
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

    # 建 meta（与原版保持一致）
    meta = {}
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): 
                continue
            d = json.loads(line)
            meta[d["utt_id"]] = {
                "label": float(d["label"]),
                "spk_id": d.get("spk_id", "NA"),
                "duration": d.get("duration", None)
            }

    # 推理
    utts, mu, var = infer_all(model, loader, device, meta)

    # 评测分支
    if args.eval == "mse":
        # 直接逐样本评测
        res = evaluate_regression(utts, mu, meta)
        report = {
            "settings": {
                "eval": "mse"
            },
            "overall": res
        }
        out_path = Path(args.out or "reg_report.json")
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        
        detail_path = out_path.with_suffix(".details.csv")
        import csv
        with open(detail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["utt_id", "true_label", "pred_score", "diff"])
            for u, p in zip(utts, mu):
                y = float(meta[u]["label"])
                writer.writerow([u, y, float(p), float(p) - y])

        print(f"Saved per-sample predictions to {detail_path}")
        return

    # ====== 保留原 win-rate 流程（兼容旧用法） ======
    pairs = sample_pairs(
        utts=utts,
        meta=meta,
        mode=args.pair_mode,
        max_pairs=args.max_pairs,
        seed=args.seed
    )
    res_overall = evaluate_winrate(
        utts=utts, mu=mu, var=var, meta=meta, pairs=pairs,
        epsilon_label=args.epsilon_label,
        tie_tol=args.tie_tol,
        use_uncertainty=args.use_uncertainty,
        z=args.z
    )

    by_spk = []
    if args.group_by_spk:
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

    report = {
        "settings": {
            "eval": "winrate",
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

    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)

    # 新增：评测类型
    ap.add_argument("--eval", type=str, default="mse", choices=["mse", "winrate"],
                    help="选择评测方式：mse（逐样本回归）或 winrate（成对方向一致率）")

    # 以下参数仅在 winrate 下有用（兼容旧流程）
    ap.add_argument("--pair-mode", type=str, default="within_spk",
                    choices=["within_spk", "cross_spk", "all_pairs"])
    ap.add_argument("--epsilon-label", type=float, default=0.05,
                    help="仅统计标签差距 >= 该阈值的样本对")
    ap.add_argument("--tie-tol", type=float, default=1e-6,
                    help="预测差距绝对值低于该阈值视为平局，不计入分母")
    ap.add_argument("--use-uncertainty", action="store_true",
                    help="若模型输出 logvar，则用 z*sqrt(var_i+var_j) 做弃权判定")
    ap.add_argument("--z", type=float, default=1.96, help="不确定性弃权的 z 分数")
    ap.add_argument("--max-pairs", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--group-by-spk", action="store_true")
    ap.add_argument("--save-pairs", action="store_true",
                    help="保存 pair 明细 CSV 便于抽查")

    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    main(args)