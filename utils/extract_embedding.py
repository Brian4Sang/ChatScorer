import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm


def single_job(sample):
    # utt_id = sample["utt_id"]
    spk_id = sample["spk_id"]
    wav_path = sample["wav_path"]

    audio, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
    feat = kaldi.fbank(audio,
                       num_mel_bins=80,
                       dither=0,
                       sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = ort_session.run(None, {
        ort_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()
    })[0].flatten().tolist()
    return spk_id, embedding


def main(args):
    # 加载 JSONL 文件为样本列表
    with open(args.jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line.strip()) for line in f]

    # 多线程提取
    all_tasks = [executor.submit(single_job, sample) for sample in samples]
    spk2embedding_list = {}

    for future in tqdm(as_completed(all_tasks), total=len(all_tasks)):
        spk_id, embedding = future.result()
        if spk_id not in spk2embedding_list:
            spk2embedding_list[spk_id] = []
        spk2embedding_list[spk_id].append(embedding)

    # 平均得到 spk2embedding
    spk2embedding = {
        spk: torch.tensor(embeds).mean(dim=0).tolist()
        for spk, embeds in spk2embedding_list.items()
    }

    # 保存结果
    torch.save(spk2embedding, f"{args.out_dir}/spk2embedding.pt")
    print(f"✅ 已保存 {len(spk2embedding)} 个说话人的 embedding 到 {args.out_dir}/spk2embedding.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True, help="输入的 JSONL 文件路径")
    parser.add_argument("--onnx_path", type=str, required=True, help="ONNX 说话人模型路径")
    parser.add_argument("--out_dir", type=str, required=True, help="保存 spk2embedding 的输出目录")
    parser.add_argument("--num_thread", type=int, default=16)
    args = parser.parse_args()

    # 初始化 ONNX 推理器
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(args.onnx_path, sess_options=option, providers=providers)

    # 初始化线程池
    executor = ThreadPoolExecutor(max_workers=args.num_thread)

    main(args)