# datasets/style_dataset.py
import json
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

class StyleDataset(Dataset):
    def __init__(self, jsonl_path, spk_embed_path, target_sr=16000, max_duration=20.0):
        """
        Args:
            jsonl_path: 训练样本列表
            spk_embed_path: 说话人 embedding 字典（.pt 文件）
            target_sr: 采样率，默认为 16kHz
            max_duration: 最长音频时长（秒）
        """
        self.data = [json.loads(line) for line in open(jsonl_path, 'r', encoding='utf-8')]
        self.spk2embedding = torch.load(spk_embed_path)
        self.target_sr = target_sr
        self.max_samples = int(max_duration * target_sr)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        wav_path = sample["wav_path"]
        spk_id = sample["spk_id"]
        label = sample["label"]

        # ✅ 加载音频
        waveform, sr = torchaudio.load(wav_path)
        assert sr == self.target_sr, f"Expected {self.target_sr}Hz but got {sr}Hz"
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform[0]  # 单声道

        # ✅ 截断或补零
        if waveform.size(0) > self.max_samples:
            waveform = waveform[:self.max_samples]
        else:
            pad = self.max_samples - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, pad))

        # ✅ 加载说话人 embedding
        spk_embedding = self.spk2embedding[spk_id]  # (192,)

        return {
            "waveform": waveform,               # (T,)
            "spk_embedding": spk_embedding,     # (192,)
            "label": torch.tensor(label, dtype=torch.float32),  # float for BCE loss
            "utt_id": sample["utt_id"]
        }