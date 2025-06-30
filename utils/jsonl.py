# prepare_jsonl_separate.py
import json
import torchaudio
from pathlib import Path
import argparse

def process_dir(wav_dir, spk_id, label, output_jsonl):
    wav_dir = Path(wav_dir)
    samples = []
    total_duration = 0.0

    for wav_path in sorted(wav_dir.glob("*.wav")):
        try:
            info = torchaudio.info(wav_path)
            duration = info.num_frames / info.sample_rate
        except Exception as e:
            print(f"✗ Failed to read {wav_path.name}: {e}")
            continue

        utt_id = f"{spk_id}_{wav_path.stem}"
        samples.append({
            "utt_id": utt_id,
            "wav_path": str(wav_path.resolve()),
            "spk_id": spk_id,
            "label": label,
            "duration": round(duration, 3)
        })

        total_duration += duration

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✓ {spk_id}: {len(samples)} samples, total duration {round(total_duration / 3600, 2)} hours")
    print(f"✓ Saved to {output_jsonl}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", required=True, help="说话人1的目录")
    parser.add_argument("--spk_id1", required=True)
    parser.add_argument("--label1", type=int, choices=[0, 1], required=True)
    parser.add_argument("--output1", required=True)

    parser.add_argument("--dir2", required=True, help="说话人2的目录")
    parser.add_argument("--spk_id2", required=True)
    parser.add_argument("--label2", type=int, choices=[0, 1], required=True)
    parser.add_argument("--output2", required=True)

    args = parser.parse_args()

    process_dir(args.dir1, args.spk_id1, args.label1, args.output1)
    process_dir(args.dir2, args.spk_id2, args.label2, args.output2)

if __name__ == "__main__":
    main()