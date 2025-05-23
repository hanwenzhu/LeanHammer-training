#!/usr/bin/bash
source /home/thomaszh/.bashrc
cd /home/thomaszh/LeanHammer-training
conda activate lm
set -xe

MODEL_PATH="/data/user_data/thomaszh/all-distilroberta-v1-lr2e-4-bs1024-nneg3-ml/final"

python retrieve_premises.py \
  --model_path $MODEL_PATH
