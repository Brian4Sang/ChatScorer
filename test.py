# debug_model_init.py
import torch
from omegaconf import OmegaConf
from models.model import ChatStyleScorer

def test_model():
    # ✅ 加载配置文件
    cfg = OmegaConf.load("configs/config.yaml")

    # ✅ 构建模型
    model = ChatStyleScorer(cfg)
    model.eval()

    # ✅ 构造伪输入
    batch_size = 2
    audio_len = 16000 * 3  # 3秒音频
    waveform = torch.randn(batch_size, audio_len)        # float32
    spk_embedding = torch.randn(batch_size, cfg.model.speaker_dim)

    # ✅ 前向推理
    with torch.no_grad():
        score = model(waveform, spk_embedding)

    print("Score shape:", score.shape)
    print("Score sample:", score)

if __name__ == "__main__":
    test_model()