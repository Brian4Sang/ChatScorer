# ChatStyleScorer Pipeline
--- For chat style tts

## Overview

This project provides a unified pipeline controlled by two variables in `run.sh`:

stage=3  
stop_stage=3

- stage: starting step number  
- stop_stage: ending step number  

---

## Environment Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- torchaudio, librosa, numpy, tqdm  
- onnxruntime (for ONNX inference)  
- yaml, jsonlines, argparse  

You must prepare the following files before running:  
pre_train/campplus.onnx  
configs/config.yaml

---

## Stage-by-Stage Description

### Stage 0 — Data Processing for WavLM
Function:  
Preprocess raw generated audios into a structured dataset format for ChatStyleScorer.  
Each audio and its metadata (speaker ID, text, waveform path, etc.) will be recorded in a JSONL file.

Inputs:  
--input-dir : directory containing generated audio files (.wav)  
--spk-id : speaker ID label used for conditioning or analysis  

Outputs:  
--output-dir : processed audio directory  
--output-jsonl : metadata file in JSON Lines format (data.jsonl)

---

### Stage 1 — Speaker Embedding Extraction
Function:  
Extract speaker embeddings for all audio samples listed in the JSONL file using the pretrained CampPlus ONNX model.  
This generates a spk2embedding.pt file mapping each speaker ID to an embedding vector.

Inputs:  
--jsonl : metadata file describing all input audios  
--onnx_path : path to the pretrained CampPlus encoder  

Outputs:  
--out_dir/spk2embedding.pt : serialized PyTorch dictionary {spk_id → embedding}

---

### Stage 3 — Training ChatStyleScorer

Function:  
Train the ChatStyleScorer model using the given configuration.

Inputs:  
--config : YAML config file specifying dataset path, model structure, and hyperparameters  

Outputs:  
Model checkpoints stored in checkpoints/, e.g.:  
checkpoints/xx/xx/  
                  ├── best.pt  
                  ├── last.pt  
└── logs/


---

### Stage 4 — Model Testing (Standard Evaluation)
Function:  
Evaluate the trained model on a validation dataset.

Inputs:  
--config : same YAML configuration as training  
--ckpt : path to the trained checkpoint  

Outputs:  
--output : JSONL file containing model predictions and scores

---

### Stage 5 — Batch Inference for ChatScore
Function:  
Perform multi-threaded inference on a large folder of generated speech outputs to compute ChatScore (style/naturalness).

Inputs:  
--root-dir : directory containing wav files to evaluate  
--max-workers : number of threads used  

Outputs:  
--output-name : JSONL file (.jsonl) containing scores

---

### Stage 6 — MSE Evaluation
Function:  
Perform pairwise evaluation of model preference predictions (A vs B) to compute winrate and MSE.

Inputs:  
--jsonl : dataset containing labeled audio pairs  
--pair-mode : defines pairing method (all_pairs, by_delta, etc.)  
--epsilon-label : margin threshold for label smoothing  
--max-pairs : number of pairs to evaluate  

Outputs:  
--out : JSON file summarizing evaluation results (winrate, MSE)


### Stage 7 — Delta-Based Pairwise Evaluation
Function:  
Evaluate preference prediction under delta-based pairing, for subjective analysis of style sensitivity.

Inputs:  
--jsonl : same validation dataset  
--pair-mode : by_delta_all  

Outputs:  
Console and log report of winrate statistics

---

## Author

Copyright © 2025  Brian   
