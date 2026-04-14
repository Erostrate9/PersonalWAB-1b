#!/usr/bin/env bash
set -euo pipefail

DEEPSPEED_INCLUDE=${DEEPSPEED_INCLUDE:-}
MASTER_PORT=${MASTER_PORT:-29500}
DATA_PATH=${DATA_PATH:-data/graph_dpo_training_data.json}
OUTPUT_DIR=${OUTPUT_DIR:-output/graph_dpo}
BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
SFT_ADAPTER_PATH=${SFT_ADAPTER_PATH:-output/graph_param/Llama-3.2-1B-Instruct/checkpoint-2280}
TRAIN_EPOCH=${TRAIN_EPOCH:-2}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
SOURCE_LENGTH=${SOURCE_LENGTH:-1024}
PROMPT_LENGTH=${PROMPT_LENGTH:-768}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
LOGGING_STEPS=${LOGGING_STEPS:-10}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-config/llama_ds_config.json}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
BETA=${BETA:-0.1}
MIN_REWARD_MARGIN=${MIN_REWARD_MARGIN:-0.05}

DEEPSPEED_CMD=(deepspeed)
if [[ -n "$DEEPSPEED_INCLUDE" ]]; then
    DEEPSPEED_CMD+=(--include="$DEEPSPEED_INCLUDE")
fi
DEEPSPEED_CMD+=(--master_port="$MASTER_PORT" dpo_llama.py)

"${DEEPSPEED_CMD[@]}" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$BASE_MODEL" \
    --model_path "$SFT_ADAPTER_PATH" \
    --train_epoch "$TRAIN_EPOCH" \
    --learning_rate "$LEARNING_RATE" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --source_length "$SOURCE_LENGTH" \
    --warmup_ratio "$WARMUP_RATIO" \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --logging_steps "$LOGGING_STEPS" \
    --deepseed_config "$DEEPSPEED_CONFIG" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --float16 \
    --train_on param \
    --prompt_length "$PROMPT_LENGTH" \
    --beta "$BETA" \
    --min_reward_margin "$MIN_REWARD_MARGIN"
