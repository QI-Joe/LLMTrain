#!/bin/bash
# ============================================================================
# Evaluation Script for Text-Only SFT Models
# ============================================================================
# Usage:
#   ./run_sft_text_only_eval.sh <experiment_dir> [cuda_device]
#
# Example:
#   ./run_sft_text_only_eval.sh ./outputs/Qwen3Gen_02-26/method_SSP_bs_2_inputdata_input_text_TextOnly_SW3
#   ./run_sft_text_only_eval.sh ./outputs/Qwen3Gen_02-26/method_SSP_bs_2_inputdata_input_text_TextOnly_SW3 1
# ============================================================================

set -e

# Default values
EXPERIMENT_DIR="${1:-}"
CUDA_DEVICE="${2:-0}"
SPLIT="${3:-test}"

# Validate
if [ -z "$EXPERIMENT_DIR" ]; then
    echo "Error: experiment_dir is required"
    echo "Usage: $0 <experiment_dir> [cuda_device] [split]"
    echo ""
    echo "Example:"
    echo "  $0 ./outputs/Qwen3Gen_02-26/method_SSP_bs_2_inputdata_input_text_TextOnly_SW3"
    echo "  $0 ./outputs/Qwen3Gen_02-26/method_SSP_bs_2_inputdata_input_text_TextOnly_SW3 1 test"
    exit 1
fi

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "Error: Experiment directory not found: $EXPERIMENT_DIR"
    exit 1
fi

echo "=============================================="
echo "Text-Only SFT Model Evaluation"
echo "=============================================="
echo "Experiment: $EXPERIMENT_DIR"
echo "CUDA Device: $CUDA_DEVICE"
echo "Split: $SPLIT"
echo "=============================================="

# Run evaluation
python src_Gen_SFTTrainer/eval_sft_text_only.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --cuda_device "$CUDA_DEVICE" \
    --split "$SPLIT" \
    --max_new_tokens 150 \
    --temperature 0.7 \
    --top_p 0.9 \
    --do_sample

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "Results saved to: ${EXPERIMENT_DIR}/eval_logs/"
echo "=============================================="
