import torchaudio
from pathlib import Path
import os

def resample(input_root, output_root, target_sr=16000, verbose=True):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for wav_path in input_root.rglob("*.wav"):
        rel_path = wav_path.relative_to(input_root)
        out_path = output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            wav, sr = torchaudio.load(wav_path)
            if sr != target_sr:
                wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)

            # ✅ 如果是多通道（如立体声），平均为单通道
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)  # shape: (1, T)

            torchaudio.save(out_path.as_posix(), wav, target_sr)

            if verbose:
                print(f"✓ {rel_path} resampled to {target_sr}Hz → {out_path}")
        except Exception as e:
            print(f"✗ Failed: {wav_path} ({e})")