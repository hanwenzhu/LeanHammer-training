#!/usr/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#SBATCH --time=2-00:00:00
#SBATCH --partition=general
# #SBATCH --time=10-00:00:00
# #SBATCH --partition=preempt

source /home/thomaszh/.bashrc
cd /home/thomaszh/LeanHammer-training
conda activate lm
set -xe

## Model arguments
MODEL_PATH="all-distilroberta-v1"
IS_SENTENCE_TRANSFORMER=True
POOLING_MODE=mean  # this has no significance if IS_SENTENCE_TRANSFORMER is True
MINI_BATCH_SIZE=32
LR=2e-4

# MODEL_PATH="all-mpnet-base-v2"
# IS_SENTENCE_TRANSFORMER=True
# POOLING_MODE=mean
# MINI_BATCH_SIZE=32
# LR=1e-4 # 1e-4 for ≠1 epoch, 7e-5 for 1 epoch

# MODEL_PATH="all-roberta-large-v1"
# IS_SENTENCE_TRANSFORMER=True
# POOLING_MODE=mean  # this has no significance if IS_SENTENCE_TRANSFORMER is True
# MINI_BATCH_SIZE=32
# LR=5e-5

# MODEL_PATH="all-MiniLM-L12-v2"
# IS_SENTENCE_TRANSFORMER=True
# POOLING_MODE=mean
# MINI_BATCH_SIZE=64
# LR=2e-4

# TODO prompt
# MODEL_PATH="l3lab/ntp-mathlib-st-deepseek-coder-1.3b"
# IS_SENTENCE_TRANSFORMER=False
# POOLING_MODE=lasttoken
# MINI_BATCH_SIZE=8

## Data arguments
DATA_DIR="/data/user_data/thomaszh/mathlib"
FILTER=True

# naive:
# DATA_DIR="/data/user_data/thomaszh/ntp-toolkit-naive/Examples/Mathlib"
# FILTER=False

NAMELESS=False  # default False

RETRIEVAL_MASK_MODE=no_positive  # none, no_positive, only_negative

## Training arguments
### current best: all-distilroberta-v1-lr2e-4-bs256-nneg3-ml-ne5
BATCH_SIZE=256
NUM_NEGATIVES_PER_STATE=3
NUM_EPOCHS=2
DTYPE=bf16  # fp16 or bf16, can check torch.cuda.is_bf16_supported(including_emulation=False)

RUN_NAME="${MODEL_PATH}-lr${LR}-bs${BATCH_SIZE}-nneg${NUM_NEGATIVES_PER_STATE}"
if [[ "$RETRIEVAL_MASK_MODE" != "none" ]]; then
    RUN_NAME+="-ml"
    if [[ "$RETRIEVAL_MASK_MODE" != "no_positive" ]]; then
        RUN_NAME+="${RETRIEVAL_MASK_MODE}"
    fi
fi
if [[ "$NUM_EPOCHS" != "1" ]]; then
    RUN_NAME+="-ne${NUM_EPOCHS}"
fi
if [[ "$DATA_DIR" == "/data/user_data/thomaszh/ntp-toolkit-naive/Examples/Mathlib" ]]; then
    RUN_NAME+="-naive"
fi
if [[ "$NAMELESS" == "True" ]]; then
    RUN_NAME+="-nameless"
fi
RUN_NAME+=""  # add any extra name
OUTPUT_DIR="/data/user_data/thomaszh/models/${RUN_NAME}"
if [[ -f "$DATA_DIR/revision" && -f "$OUTPUT_DIR/final/revision" && "$(cat $DATA_DIR/revision)" != "$(cat $OUTPUT_DIR/final/revision)" ]]; then
    RESUME_FROM_CHECKPOINT="False"
    rm -rf $OUTPUT_DIR
    echo "Training on data at $(cat $DATA_DIR/revision); overriding previous checkpoint trained on earlier data at $(cat $OUTPUT_DIR/final/revision)"
elif compgen -G "$OUTPUT_DIR/checkpoint-*"; then
    RESUME_FROM_CHECKPOINT="True"  # note: the train_args.resume_from_checkpoint argument expects a `str`, see `train.py` for explicit conversion to bool
    echo "Resuming from latest checkpoint at $OUTPUT_DIR"
else
    RESUME_FROM_CHECKPOINT="False"
    echo "Starting training from scratch at $OUTPUT_DIR"
fi

python train.py \
    --model_name_or_path $MODEL_PATH \
    --is_sentence_transformer $IS_SENTENCE_TRANSFORMER \
    --pooling_mode $POOLING_MODE \
    --mini_batch_size $MINI_BATCH_SIZE \
    \
    --data_dir $DATA_DIR \
    --filter $FILTER \
    --nameless $NAMELESS \
    --retrieval_mask_mode $RETRIEVAL_MASK_MODE \
    --num_negatives_per_state $NUM_NEGATIVES_PER_STATE \
    \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 64 \
    --learning_rate $LR \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --batch_sampler "batch_sampler" \
    --$DTYPE True \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 1.0 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    \
    --output_dir $OUTPUT_DIR \
    --eval_strategy "steps" \
    --eval_steps 0.01 \
    --save_strategy "steps" \
    --save_steps 0.01 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --report_to "all" \
    --run_name $RUN_NAME \
    --resume_from_checkpoint $RESUME_FROM_CHECKPOINT \
    # --load_best_model_at_end True \

# Note:
# * gradient checkpointing or zero stage3 offloading might be? incompatible with gradcache TODO check
