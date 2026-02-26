#!/bin/bash
# SFTTrainer Training Script for Generation Task
# Usage: bash run_sft_train.sh [GPU_ID] [TOPIC_NAME]
#
# Output folder structure (matches ZGeneration):
#   outputs/{model}Gen_{date}/method_{method}_bs_{bs}_inputdata_{prompt_key}_{topic}/
#     ├── checkpoints/
#     │   └── final_model/
#     ├── eval_logs/
#     ├── logs/
#     └── config.json

GPU_ID=${1:-2}
TOPIC_NAME=${2:-"SFT_SSL"}
MODEL="Qwen3-4B-Instruct-2507"  # Or "Llama-3.3-8B-Instruct"

echo "=============================================="
echo "SFTTrainer Generation Training"
echo "GPU: $GPU_ID  (CUDA_VISIBLE_DEVICES=$GPU_ID → visible as cuda:0 inside Python)"
echo "Model: $MODEL"
echo "Topic: $TOPIC_NAME"
echo "=============================================="

cd "$(dirname "$0")/.."

# Prepare log file (timestamped, in logs/ at workspace root)
LOG_DIR="logs/sft_train"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${MODEL}_gpu${GPU_ID}_$(date '+%m%d_%H%M%S').log"
echo "Log file: $LOG_FILE"

# Set CUDA_VISIBLE_DEVICES HERE, before Python starts, so libcuda is
# initialised with only this GPU visible.  The in-process env-var trick
# in train_sft.py is kept as a safety net but cannot beat shell-level isolation.
CUDA_VISIBLE_DEVICES=$GPU_ID python -m src_Gen_SFTTrainer.train_sft \
    --model_name "$MODEL" \
    --cuda_device "$GPU_ID" \
    --data_path "./data" \
    --output_dir "./outputs" \
    --max_seq_length 2183 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --gradient_accumulation_steps 2 \
    --warmup_ratio 0.1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --topic_name "$TOPIC_NAME" \
    --seed 42 \
    --val_ratio 0.004 \
    --test_ratio 0.004 \
    --few_shot \
    --shots_per_class 4 2>&1 | tee "$LOG_FILE"
    # --semi_supervised \
    # --semi_ratio 0.1 \

echo "Training completed!"
