#!/usr/bin/env bash
set -euo pipefail

TASK_FILE=${TASK_FILE:-data/user_instructions.json}
USER_HISTORY_FILE=${USER_HISTORY_FILE:-data/user_history.json}
USER_PROFILE_FILE=${USER_PROFILE_FILE:-data/user_profiles.json}
ALL_PRODUCTS_FILE=${ALL_PRODUCTS_FILE:-data/all_products.json}
OUTPUT_FILE=${OUTPUT_FILE:-data/graph_dpo_training_data.json}
MEM_TOKEN_LENGTH=${MEM_TOKEN_LENGTH:-768}
REWARD_MARGIN=${REWARD_MARGIN:-0.05}
TOKENIZER_NAME=${TOKENIZER_NAME:-meta-llama/Llama-3.2-1B-Instruct}

python prepare_graph_dpo_data.py \
    --task_file "$TASK_FILE" \
    --user_history_file "$USER_HISTORY_FILE" \
    --user_profile_file "$USER_PROFILE_FILE" \
    --all_products_file "$ALL_PRODUCTS_FILE" \
    --output_file "$OUTPUT_FILE" \
    --mem_token_length "$MEM_TOKEN_LENGTH" \
    --reward_margin "$REWARD_MARGIN" \
    --tokenizer_name "$TOKENIZER_NAME"
