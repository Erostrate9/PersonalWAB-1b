#!/usr/bin/env bash
set -euo pipefail

DEEPSPEED_INCLUDE=${DEEPSPEED_INCLUDE:-}
MASTER_PORT=${MASTER_PORT:-29999}
PARAM_DATA_PATH=${PARAM_DATA_PATH:-data/graph_param_data.json}
FUNCTION_DATA_PATH=${FUNCTION_DATA_PATH:-data/graph_function_data.json}
OUTPUT_DIR=${OUTPUT_DIR:-output/graph_param}
MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-3.2-1B-Instruct}
TRAIN_EPOCH=${TRAIN_EPOCH:-2}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
SOURCE_LENGTH=${SOURCE_LENGTH:-1280}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
LOGGING_STEPS=${LOGGING_STEPS:-10}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-config/llama_ds_config.json}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
TEMPERATURE=${TEMPERATURE:-1.0}
TRAIN_ON=${TRAIN_ON:-function_param}
TASK_WEIGHTS=${TASK_WEIGHTS:-'{"recommend": 2}'}

deepspeed --include="$DEEPSPEED_INCLUDE" --master_port="$MASTER_PORT" finetune_llama.py \
    --data_path "$PARAM_DATA_PATH" \
    --function_data_path "$FUNCTION_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
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
    --temperature "$TEMPERATURE" \
    --float16 \
    --train_on "$TRAIN_ON" \
    --task_weights "$TASK_WEIGHTS"
