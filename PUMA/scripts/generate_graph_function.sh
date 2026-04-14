#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-output/graph_param/Llama-3.2-1B-Instruct/checkpoint-2280}
BASE_MODEL=${BASE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
DATA_PATH=${DATA_PATH:-data/graph_function_data.json}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
RESULT_FILE=${RESULT_FILE:-output/res/graph_function_res.json}
NUM_PROCESSES=${NUM_PROCESSES:-1}
NUM_MACHINES=${NUM_MACHINES:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
DYNAMO_BACKEND=${DYNAMO_BACKEND:-no}

accelerate launch \
    --num_processes "$NUM_PROCESSES" \
    --num_machines "$NUM_MACHINES" \
    --mixed_precision "$MIXED_PRECISION" \
    --dynamo_backend "$DYNAMO_BACKEND" \
    test_llama.py \
    --model_path "$MODEL_PATH" \
    --base_model "$BASE_MODEL" \
    --data_path "$DATA_PATH" \
    --bf16 \
    --test_on function \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --res_file "$RESULT_FILE"
