import argparse
import torch
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spk2info", type=str, default="/brian/cosy/cosyvoice/CosyVoice/ChatStyleScorer/pre_train/spk2info.pt",help="已有的 spk2info.pt 文件路径")
    parser.add_argument("--spk2embedding", type=str,  default="/brian/cosy/cosyvoice/CosyVoice/ChatStyleScorer/data/banshu/spk2embedding.pt",help="要追加的 spk2embedding.pt 文件路径")
    parser.add_argument("--overwrite", action="store_true", help="是否允许覆盖已有说话人")
    args = parser.parse_args()

    if not os.path.exists(args.spk2info):
        print(f"找不到 spk2info 文件: {args.spk2info}")
        return
    if not os.path.exists(args.spk2embedding):
        print(f"找不到 spk2embedding 文件: {args.spk2embedding}")
        return

    spk2info = torch.load(args.spk2info)
    new_spk2embedding = torch.load(args.spk2embedding)

    # 追加spk
    added, skipped, overwritten = 0, 0, 0
    for spk_id, emb in new_spk2embedding.items():
        if spk_id in spk2info:
            if args.overwrite:
                overwritten += 1
            else:
                skipped += 1
                continue
        spk2info[spk_id] = {"embedding": torch.tensor(emb)}
        added += 1

    # 保存回原路径
    torch.save(spk2info, args.spk2info)
    print(f"已追加 {added} 个，说话人（跳过 {skipped} 个，覆盖 {overwritten} 个）")
    print(f"已保存到: {args.spk2info}")

if __name__ == "__main__":
    main()