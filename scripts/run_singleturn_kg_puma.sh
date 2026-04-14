#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${ENV_NAME:-pwab}
MODEL=${MODEL:-finetune/llama}
USER_MODE=${USER_MODE:-no}
USER_MODEL=${USER_MODEL:-gpt-4o-mini}
TASK_SPLIT=${TASK_SPLIT:-test}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-1}
MAX_STEPS=${MAX_STEPS:--1}
END_INDEX=${END_INDEX:--1}
PUMA_GENERATE=${PUMA_GENERATE:-1}
PUMA_MODEL_PATH=${PUMA_MODEL_PATH:-PUMA/output/graph_param/Llama-3.2-1B-Instruct/checkpoint-2280}
PUMA_BASE_MODEL=${PUMA_BASE_MODEL:-meta-llama/Llama-3.2-1B-Instruct}
PUMA_FUNCTION_FILE=${PUMA_FUNCTION_FILE:-PUMA/output/res/graph_function_res.json}
PUMA_PARAM_FILE=${PUMA_PARAM_FILE:-PUMA/output/res/graph_param_res.json}
MEM_TOKEN_LENGTH=${MEM_TOKEN_LENGTH:-768}

python run.py \
    --env "$ENV_NAME" \
    --model "$MODEL" \
    --user_mode "$USER_MODE" \
    --user_model "$USER_MODEL" \
    --agent_strategy kg_puma \
    --task_split "$TASK_SPLIT" \
    --max_concurrency "$MAX_CONCURRENCY" \
    --max_steps "$MAX_STEPS" \
    --end_index "$END_INDEX" \
    --puma_generate "$PUMA_GENERATE" \
    --puma_model_path "$PUMA_MODEL_PATH" \
    --puma_base_model "$PUMA_BASE_MODEL" \
    --puma_function_file "$PUMA_FUNCTION_FILE" \
    --puma_param_file "$PUMA_PARAM_FILE" \
    --mem_token_length "$MEM_TOKEN_LENGTH"
