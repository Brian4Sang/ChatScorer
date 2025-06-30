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

        # ✅ 1. 加载预训练 WavLM，并冻结参数（只提取语音特征）
        self.wavlm = WavLMModel.from_pretrained(model_cfg.wavlm_model_name)
        for param in self.wavlm.parameters():
            param.requires_grad = False

        # ✅ 2. Transformer encoder 层（用于建模说话节奏、停顿、语调）
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

        # ✅ 3. Pooling：默认 mean，可选 attention
        self.pooling_type = model_cfg.get('pooling_type', 'mean')
        if self.pooling_type == 'attention':
            self.attn_pool = nn.Sequential(
                nn.Linear(model_cfg.transformer_dim, 1)
            )

        # ✅ 4. 说话人 embedding 映射投影（用于做残差）
        self.spk_proj = nn.Linear(model_cfg.speaker_dim, model_cfg.transformer_dim)

        # ✅ 5. 打分头（score head）：MLP 输出一个标量分数
        self.score_head = nn.Sequential(
            nn.Linear(model_cfg.transformer_dim, model_cfg.score_head),
            nn.ReLU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(model_cfg.score_head, model_cfg.output_dim)  # 输出 raw score，可接 sigmoid
        )

    def forward(self, waveform: torch.Tensor, spk_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) float tensor, 单声道 16kHz PCM 语音
            spk_embedding: (B, D_spk) float tensor, 每人一个 embedding

        Returns:
            score: (B,) 打分结果（raw score，可接 sigmoid）
        """
        assert waveform.dim() == 2, "waveform shape must be (B, T)"
        assert spk_embedding.dim() == 2, "spk_embedding shape must be (B, D)"

        # ✅ 1. 提取帧级语音特征
        with torch.no_grad():
            features = self.wavlm(waveform).last_hidden_state  # (B, T', D)

        # ✅ 2. Transformer 编码建模时序
        encoded = self.transformer(features)  # (B, T', D)

        # ✅ 3. Pooling：将 (B, T', D) → (B, D)
        if self.pooling_type == 'mean':
            pooled_feat = reduce(encoded, 'b t d -> b d', 'mean')
        elif self.pooling_type == 'attention':
            attn_weights = self.attn_pool(encoded).squeeze(-1)  # (B, T)
            attn_weights = F.softmax(attn_weights, dim=-1)
            pooled_feat = torch.sum(encoded * attn_weights.unsqueeze(-1), dim=1)  # (B, D)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling_type}")

        # ✅ 4. 减去 speaker embedding（做残差去音色）
        spk_feat = self.spk_proj(spk_embedding)  # (B, D)
        style_feat = pooled_feat - spk_feat      # (B, D)

        # ✅ 5. 打分
        score = self.score_head(style_feat).squeeze(-1)  # (B,)

        return score