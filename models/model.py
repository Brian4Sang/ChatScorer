# models/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel
from einops import reduce
from omegaconf import DictConfig

class ChatStyleScorer(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        model_cfg = cfg.model

        #  1. 加载预训练 WavLM，并冻结参数（只提取语音特征）
        self.wavlm = WavLMModel.from_pretrained(model_cfg.wavlm_model_name)
        for param in self.wavlm.parameters():
            param.requires_grad = False

        #  2. Transformer encoder 层（用于建模说话节奏、停顿、语调）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_cfg.transformer_dim,
            nhead=model_cfg.transformer_heads,
            dim_feedforward=4 * model_cfg.transformer_dim,
            dropout=model_cfg.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_cfg.transformer_layers
        )

        #  3. Pooling：默认 mean，可选 attention
        self.pooling_type = model_cfg.get('pooling_type', 'mean')
        if self.pooling_type == 'attention':
            self.attn_pool = nn.Sequential(
                nn.Linear(model_cfg.transformer_dim, 1)
            )

        #  4. 说话人 embedding 映射投影（用于做残差）
        self.spk_proj = nn.Linear(model_cfg.speaker_dim, model_cfg.transformer_dim)

        #  5. 打分头（score head）：MLP 输出一个标量分数
        self.score_head = nn.Sequential(
            nn.Linear(model_cfg.transformer_dim, model_cfg.score_head),
            nn.ReLU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(model_cfg.score_head, model_cfg.output_dim)  # 输出 raw score，可接 sigmoid
        )

   # models/model.py

    def forward(self, waveform: torch.Tensor, spk_embedding: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        waveform: (B, T)
        spk_embedding: (B, D)
        attention_mask: (B, T), dtype=bool, True 表示有效帧
        """
        with torch.no_grad():
            outputs = self.wavlm(waveform, attention_mask=attention_mask)
            features = outputs.last_hidden_state  # (B, T', D)

        # frame-level 掩码
        frame_mask = attention_mask[:, ::320]
        if frame_mask.shape[1] > features.shape[1]:
            frame_mask = frame_mask[:, :features.shape[1]]
        elif frame_mask.shape[1] < features.shape[1]:
            # pad False（无效）到右侧，确保长度对齐
            pad_len = features.shape[1] - frame_mask.shape[1]
            frame_mask = F.pad(frame_mask, (0, pad_len), value=False)  
            
        encoded = self.transformer(features, src_key_padding_mask=~frame_mask)

        #  pooling 阶段：masked mean
        if self.pooling_type == 'mean':
            frame_mask_f = frame_mask.unsqueeze(-1).type_as(encoded)  # (B, T, 1)
            summed = torch.sum(encoded * frame_mask_f, dim=1)             # 有效帧求和
            lens = frame_mask_f.sum(dim=1).clamp(min=1e-6)                # 避免除以0
            pooled_feat = summed / lens                                       # (B, D)
        elif self.pooling_type == 'attention':
            attn_weights = self.attn_pool(encoded).squeeze(-1)  # (B, T)
            attn_weights = attn_weights.masked_fill(~attention_mask, -1e9)
            attn_weights = torch.softmax(attn_weights, dim=-1)  # (B, T)
            pooled_feat = torch.sum(encoded * attn_weights.unsqueeze(-1), dim=1)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        # 残差去音色 + MLP 得分
        spk_feat = self.spk_proj(spk_embedding)
        style_feat = pooled_feat - spk_feat
        score = self.score_head(style_feat).squeeze(-1)
        return score