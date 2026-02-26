#!/bin/bash
# SFTTrainer Evaluation Script
# Usage: bash run_sft_eval.sh [GPU_ID] [EXPERIMENT_DIR]
#
# Mode A (recommended): provide --experiment_dir
#   adapter path, model name, seq length, and exact test split are
#   all auto-resolved from the saved config.json and eval_logs/ inside.
#
# Mode B (legacy): omit --experiment_dir and set data/split flags manually.

GPU_ID=${1:-0}
EXPERIMENT_DIR=${2:-"outputs/Qwen3Gen_02-25/method_FSL_bs_2_inputdata_input_text_SFT_SSL"}

echo "=============================================="
echo "SFTTrainer Generation Evaluation"
echo "GPU:            $GPU_ID"
echo "Experiment dir: $EXPERIMENT_DIR"
echo "=============================================="

cd "$(dirname "$0")/.."

python -m src_Gen_SFTTrainer.eval_sft \
    --cuda_device "$GPU_ID" \
    --experiment_dir "$EXPERIMENT_DIR" \
    --split "test" \
    --max_new_tokens 150 \
    --temperature 0.7 \
    --top_p 0.9 \
    --do_sample \
    --seed 42

echo "Evaluation completed!"
