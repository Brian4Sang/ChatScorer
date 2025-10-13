#!/bin/bash
# Copyright 2025 Brian. All Rights Reserved.

stage=5
stop_stage=5

data_url=www.openslr.org/resources/60
data_dir=/mnt/lyuxiang.lx/data/tts/openslr/libritts
pretrained_model_dir=pre_train/

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data Process for WavLM"
    python utils/processor.py \
    --input-dir /brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-new/test/female/dpo-test/text_149 \
    --output-dir data/wave/test/female/dpo/text_149 \
    --spk-id tianqing \
    --output-jsonl data/wave/test/female/dpo/text_149/data.jsonl \
    --skip-channel 

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt"
    python utils/extract_embedding.py \
    --jsonl data/hotshan/zh_male_yuanboxiaoshu_moon_bigtts/data.jsonl \
    --out_dir data/hotshan/zh_male_yuanboxiaoshu_moon_bigtts \
    --onnx_path $pretrained_model_dir/campplus.onnx
  # python tools/extract_embedding_dpo.py \
  #   --root_dir /brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs/test/male/A-ori-zero \
  #   --onnx_path $pretrained_model_dir/campplus.onnx \
  #   --spk_emb_path $pretrained_model_dir/pre_models/spk2info.pt
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "start trainging for chatstyle..."
    python train.py \
    --config configs/config.yaml \

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "start testing for chatstyle..."
    python test.py \
    --config configs/config.yaml \
    --ckpt checkpoints/re/v0.2/best.pt \
    --output data/wave/test/test-val.jsonl
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "start testing-n for chatstylesss..."
    python testn.py \
      --config configs/config.yaml \
      --ckpt checkpoints/re/v0.2-ext/best.pt \
      --root-dir /brian/cosy/cosyvoice/CosyVoice/examples/libritts/cosyvoice2/outputs-sft/new/rank3-null-csc-all/tianqing \
      --max-workers 8 \
      --output-name chatScore.jsonl 
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "start testing-n for chatstyle..."
    python testmse.py \
      --config configs/config.yaml \
      --ckpt checkpoints/re/v0.2-ext/best.pt \
      --jsonl data/curated/jsonl/spilt/unio/val.jsonl \
      --pair-mode all_pairs \
      --epsilon-label 0.1 \
      --max-pairs 1000\
      --eval winrate \
      --out logs/test/v0.4_winrate_report.json
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "start testing-n for chatstyle..."
    python testwinrate.py \
      --config configs/config.yaml \
      --ckpt checkpoints/re/v0.2-ext/best.pt \
      --jsonl data/curated/jsonl/spilt/unio/val.jsonl \
      --eval winrate \
      --pair-mode by_delta_all \
      --epsilon-label 0.1 \
      --max-pairs 1000
fi