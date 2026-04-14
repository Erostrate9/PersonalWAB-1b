#!/usr/bin/env bash
set -euo pipefail

TASK_FILE=${TASK_FILE:-PersonalWAB/envs/pwab/data/user_instructions.json}
USER_HISTORY_FILE=${USER_HISTORY_FILE:-PersonalWAB/envs/pwab/data/user_history_part_*.json}
USER_PROFILE_FILE=${USER_PROFILE_FILE:-PersonalWAB/envs/pwab/data/user_profiles.json}
ALL_PRODUCTS_FILE=${ALL_PRODUCTS_FILE:-data/all_products.json}
FUNCTION_FILE=${FUNCTION_FILE:-PUMA/output/res/graph_function_res.json}
PARAM_FILE=${PARAM_FILE:-PUMA/output/res/graph_param_res_train.json}
DPO_OUTPUT=${DPO_OUTPUT:-PUMA/data/graph_dpo_scores.json}

python test.py \
    --graph_mode \
    --evaluate_dpo True \
    --task_file "$TASK_FILE" \
    --user_history_file "$USER_HISTORY_FILE" \
    --user_profile_file "$USER_PROFILE_FILE" \
    --all_products "$ALL_PRODUCTS_FILE" \
    --function_file "$FUNCTION_FILE" \
    --param_file "$PARAM_FILE" \
    --dpo_output "$DPO_OUTPUT"
