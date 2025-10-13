# # datasets/style_dataset.py
import json
import torch
import random
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

def load_json(path):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, dict):
            # 顶层是字典就当作一条样本
            return [obj]
        else:
            raise ValueError(f"不支持的JSON顶层类型: {type(obj)} @ {path}")
    else:
        # 其它一律按 jsonl 逐行读取
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    items.append(json.loads(s))
                except Exception:
                    # 兼容行尾逗号
                    items.append(json.loads(s.rstrip(",")))
        return items

class StyleDataset(Dataset):
    def __init__(self, cfg, split="train"):
        """
        cfg: OmegaConf 对象，包含 dataset 相关字段
        split: "train" / "val" / "test"
        """
        self.data_cfg = cfg.dataset
        jsonl_paths = self.data_cfg.jsonl_path[split]
        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]

        self.data = []
        for path in jsonl_paths:
            self.data.extend(load_json(path))

        self.spk2embedding = torch.load(self.data_cfg.spk_embedding_path, weights_only=False)
        # embedding = self.spk2info[spk_id]['embedding']
        # self.max_samples = int(self.data_cfg.max_duration * self.data_cfg.target_sr)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        wav_path = sample["wav_path"]
        spk_id = sample["spk_id"]
        label = float(sample["label"]) if "label" in sample else None
        utt_id = sample["utt_id"]
        # duration = sample["duration"]

        #  长音频切片
        chunk_samples = int(self.data_cfg.chunk_size * self.data_cfg.target_sr)
        hop_samples = int(self.data_cfg.chunk_hop * self.data_cfg.target_sr)
        
        waveform, sr = torchaudio.load(wav_path)

        #  如果多通道，取平均为单通道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform[0]

        #  如果采样率不一致，重采样为目标采样率
        if sr != self.data_cfg.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.data_cfg.target_sr)
            waveform = resampler(waveform)  # shape (T,) → (T')

        if waveform.size(0) > chunk_samples:
            max_start = waveform.size(0) - chunk_samples
            start = random.randint(0, max_start // hop_samples) * hop_samples
            waveform = waveform[start: start + chunk_samples]
            
        if spk_id not in self.spk2embedding:
            raise KeyError(f"Missing spk embedding for '{spk_id}'")

        spk_embedding = self.spk2embedding[spk_id]['embedding']  # (D,)
        
        return_dict = {
            "waveform": waveform,
            "spk_embedding": spk_embedding,
            "utt_id": utt_id,
        }
        if label is not None:
            return_dict["label"] = torch.tensor(label, dtype=torch.float32)

        return return_dict

# datasets/style_dataset.py
# import json
# import torch
# import random
# import torchaudio
# from torch.utils.data import Dataset
# from pathlib import Path

# class StyleDataset(Dataset):
#     def __init__(self, cfg, split="train"):
#         """
#         cfg: OmegaConf 对象，包含 dataset 相关字段
#         split: "train" / "val" / "test"
#         """
#         self.data_cfg = cfg.dataset
#         jsonl_paths = self.data_cfg.jsonl_path[split]
#         if isinstance(jsonl_paths, str):
#             jsonl_paths = [jsonl_paths]

#         self.data = []
#         for path in jsonl_paths:
#             with open(path, 'r', encoding='utf-8') as f:
#                 for line in f:
#                     sample = json.loads(line)
#                     self.data.append(sample)

#         self.spk2embedding = torch.load(self.data_cfg.spk_embedding_path, weights_only=False)
#         # embedding = self.spk2info[spk_id]['embedding']
#         # self.max_samples = int(self.data_cfg.max_duration * self.data_cfg.target_sr)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         wav_path = sample["wav_path"]
#         spk_id = sample["spk_id"]
#         label = float(sample["label"])
#         utt_id = sample["utt_id"]
#         # duration = sample["duration"]

#         #  长音频切片
#         chunk_samples = int(self.data_cfg.chunk_size * self.data_cfg.target_sr)
#         hop_samples = int(self.data_cfg.chunk_hop * self.data_cfg.target_sr)
        
#         waveform, sr = torchaudio.load(wav_path)

#         #  如果多通道，取平均为单通道
#         if waveform.shape[0] > 1:
#             waveform = waveform.mean(dim=0)
#         else:
#             waveform = waveform[0]

#         #  如果采样率不一致，重采样为目标采样率
#         if sr != self.data_cfg.target_sr:
#             resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.data_cfg.target_sr)
#             waveform = resampler(waveform)  # shape (T,) → (T')

#         if waveform.size(0) > chunk_samples:
#             max_start = waveform.size(0) - chunk_samples
#             start = random.randint(0, max_start // hop_samples) * hop_samples
#             waveform = waveform[start: start + chunk_samples]
            
#         if spk_id not in self.spk2embedding:
#             raise KeyError(f"Missing spk embedding for '{spk_id}'")

#         spk_embedding = self.spk2embedding[spk_id]['embedding']  # (D,)
#         # print(f"loading...............{spk_id} and its label : {label}")

#         return {
#             "waveform": waveform,
#             "spk_embedding": spk_embedding,
#             "label": torch.tensor(label, dtype=torch.float32),
#             "utt_id": utt_id,
#             # "duration": duration
#         }