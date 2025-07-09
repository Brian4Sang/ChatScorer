#!/bin/bash
# Copyright 2025 Brian. All Rights Reserved.

stage=4
stop_stage=4

data_url=www.openslr.org/resources/60
data_dir=/mnt/lyuxiang.lx/data/tts/openslr/libritts
pretrained_model_dir=pre_train/

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data Process for WavLM"
    python utils/processor.py \
    --input-dir /brian/share/tts_data/bantong/audio \
    --output-dir /brian/cosy/cosyvoice/CosyVoice/ChatStyleScorer/data/wave/test/male/ori/text_1026 \
    --spk-id bantong \
    --label 0 \
    --output-jsonl /brian/cosy/cosyvoice/CosyVoice/ChatStyleScorer/data/bantong/data.jsonl \
    --skip-channel \
    --max-num 500
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt"
    python utils/extract_embedding.py \
    --jsonl data/bantong/data.jsonl \
    --out_dir data/bantong \
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
    --ckpt checkpoints/best.pt \
    --output outputs/test01.jsonl
fi