import os
import json
import torch
import torchaudio
import soundfile as sf
import argparse
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_audio(wav_path, output_dir, spk_id, label, target_sr, max_samples,
                  skip_resample=False, skip_channel=False):
    try:
        wav_path = Path(wav_path)
        waveform, sr = torchaudio.load(str(wav_path))  # waveform: (C, T)

        # 转为单通道
        if not skip_channel:
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform[0]
        else:
            waveform = waveform[0] if waveform.dim() == 2 else waveform

        # ✅ 重采样
        if not skip_resample and sr != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)
        else:
            target_sr = sr  # 用原采样率更新 target_sr

        # # ✅ 裁剪或补零
        # if waveform.size(0) > max_samples:
        #     waveform = waveform[:max_samples]
        # else:
        #     pad = max_samples - waveform.size(0)
        #     waveform = torch.nn.functional.pad(waveform, (0, pad))

        duration = waveform.size(0) / target_sr
        utt_id = f"{wav_path.stem}"
        save_path = Path(output_dir) / f"{utt_id}.wav"
        sf.write(str(save_path), waveform.numpy(), samplerate=target_sr)

        return {
            "utt_id": utt_id,
            "wav_path": str(save_path.resolve()),
            "spk_id": spk_id,
            "label": label,
            "duration": round(duration, 3)
        }

    except Exception as e:
        print(f"✗ Error in {wav_path.name}: {e}")
        return None

def collect_wavs(input_dir):
    return list(Path(input_dir).rglob("*.wav"))

def run_processing(args):
    os.makedirs(args.output_dir, exist_ok=True)
    wav_paths = collect_wavs(args.input_dir)
    wav_paths = collect_wavs(args.input_dir)
    if args.max_num is not None:
        wav_paths = wav_paths[:args.max_num]
    max_samples = int(args.max_duration * args.target_sr)
    results = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [
            executor.submit(
                process_audio,
                wav_path,
                args.output_dir,
                args.spk_id,
                args.label,
                args.target_sr,
                max_samples,
                args.skip_resample,
                args.skip_channel
            )
            for wav_path in wav_paths
        ]

        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = f.result()
            if result:
                results.append(result)

    # 保存 jsonl 文件
    os.makedirs(Path(args.output_jsonl).parent, exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    total_dur = sum(item["duration"] for item in results)
    print(f"\n✅ Processed {len(results)} files | Total duration: {round(total_dur / 3600, 2)} hours")
    print(f"✅ Saved jsonl to {args.output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="输入原始音频目录")
    parser.add_argument("--output-dir", required=True, help="输出处理后音频目录")
    parser.add_argument("--spk-id", required=True, help="说话人 ID")
    parser.add_argument("--label", type=int, choices=[0, 1], required=True, help="样本标签（0:合成，1:真实）")
    parser.add_argument("--output-jsonl", required=True, help="保存的 jsonl 路径")
    parser.add_argument("--target-sr", type=int, default=16000, help="目标采样率")
    parser.add_argument("--max-duration", type=float, default=10.0, help="最大音频时长（秒）")
    parser.add_argument("--num-workers", type=int, default=8, help="并行线程数")
    parser.add_argument("--skip-resample", action="store_true", help="跳过采样率处理")
    parser.add_argument("--skip-channel", action="store_true", help="跳过通道数处理")
    parser.add_argument("--max-num", type=int, default=500, help="最多处理的音频条数（默认不限制）")
    args = parser.parse_args()

    run_processing(args)