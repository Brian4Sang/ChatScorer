# datasets/collate.py
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_style_batch(batch):
    """
    统一 collate 函数，兼容训练/验证/推理三种情况。
    """
    waveforms = [item["waveform"] for item in batch]
    lengths = [w.shape[0] for w in waveforms]
    utt_ids = [item["utt_id"] for item in batch]

    # waveform padding
    waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)  # (B, T)
    max_len = waveforms.shape[1]
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    for i, l in enumerate(lengths):
        attention_mask[i, :l] = 1

    batch_dict = {
        "waveforms": waveforms,
        "utt_ids": utt_ids,
        "attention_mask": attention_mask
    }

    # 训练/验证时存在 spk_embedding 和 label
    if "spk_embedding" in batch[0]:
        spk_embeddings = [item["spk_embedding"] for item in batch]
        batch_dict["spk_embeddings"] = torch.stack(spk_embeddings)

    if "label" in batch[0]:
        labels = [item["label"] for item in batch]
        batch_dict["labels"] = torch.stack(labels)

    return batch_dict

# # datasets/collate.py

# import torch
# from torch.nn.utils.rnn import pad_sequence

# def collate_style_batch(batch):
#     """
#     Args:
#         batch: List of dicts with keys:
#             - waveform: Tensor (T,)
#             - spk_embedding: Tensor (D,)
#             - label: FloatTensor scalar
#             - utt_id: str

#     Returns:
#         A dict with batched tensors:
#             - waveforms: (B, T) (zero-padded)
#             - spk_embeddings: (B, D)
#             - labels: (B,)
#             - utt_ids: list[str]
#     """
#     waveforms = [item["waveform"] for item in batch]
#     lengths = [w.shape[0] for w in waveforms]
#     spk_embeddings = [item["spk_embedding"] for item in batch]
#     labels = [item["label"] for item in batch]
#     utt_ids = [item["utt_id"] for item in batch]

#     spk_embeddings = torch.stack(spk_embeddings)  # (B, D)
#     labels = torch.stack(labels)  # (B,)
    
#     # pad waveform to max length in batch
#     waveforms = pad_sequence(waveforms, batch_first=True, padding_value=0.0)  # (B, T)
#     max_len = waveforms.shape[1]
#     attention_mask = torch.zeros((len(batch),max_len),dtype=torch.bool)
#     for i,l in enumerate(lengths):
#         attention_mask[i, :l] = 1

#     return {
#         "waveforms": waveforms,
#         "spk_embeddings": spk_embeddings,
#         "labels": labels,
#         "utt_ids": utt_ids,
#         "attention_mask": attention_mask
#     }