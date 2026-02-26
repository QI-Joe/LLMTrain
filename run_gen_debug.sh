#!/usr/bin/env bash
# Debug Run for Generation Task - Raw Model Evaluation
# Runs multiple evaluations concurrently on different GPUs

set -euo pipefail
export PYTHONUNBUFFERED=1

cd "$(dirname "$0")"
mkdir -p logs

ts() { date '+%F %T'; }

pids=()
names=()

start() {
  local name="$1"; shift
  echo "$(ts) Starting $name ..."
  "$@" >"logs/${name}.out" 2>&1 &
  pids+=($!)
  names+=("$name")
}

TIMESTAMP=$(date '+%m-%d_%H-%M')
echo "Starting Generation Task Raw Model Evaluation..."
echo "Timestamp: $TIMESTAMP"

# Task 1: Qwen3-4B Raw Model Test (GPU 0)
start Qwen4b_rawModel_test_${TIMESTAMP} \
env CUDA_VISIBLE_DEVICES=0 python3 ZGeneration/predict_gen.py \
    --model_name "Qwen3-4B-Instruct-2507" \
    --cuda_device 0 \
    --topic_name "Qwen3_test_02" \
    --experiment_name "Multi_Run_${TIMESTAMP}" \
    --few_shot \
    --shots_per_class 16 \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_epochs 4 \
    --learning_rate 1e-5 \
    --max_seq_length 2183 \
    --max_new_tokens 20 \
    --val_ratio 0.1 \
    --test_ratio 0.2 \
    --checkpoint_dir outputs/Qwen3Gen_02-25/method_FSL_bs_2_inputdata_input_text_SFT_SSL/checkpoints/final_model \
    # --raw_model \

# Task 2: Llama3.1-8B Raw Model Test (GPU 1)
start Llama31_rawModel_test_${TIMESTAMP} \
env CUDA_VISIBLE_DEVICES=1 python3 ZGeneration/predict_gen.py \
    --model_name "llama3.1-8B-Instruct" \
    --cuda_device 1 \
    --topic_name "llama31_test_02" \
    --experiment_name "Multi_Run_${TIMESTAMP}" \
    --few_shot \
    --shots_per_class 16 \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_epochs 4 \
    --learning_rate 1e-5 \
    --max_seq_length 2183 \
    --max_new_tokens 20 \
    --val_ratio 0.1 \
    --test_ratio 0.2 \
    --checkpoint_dir outputs/llama3.1Gen_02-24/method_FSL_bs_2_inputdata_input_text_Llama3.1_FSL16/checkpoints/checkpoint-epoch-3 \

# Handle Ctrl-C/TERM gracefully
trap 'echo; echo "Stopping all jobs..."; kill "${pids[@]}" 2>/dev/null || true' INT TERM

# Wait for all jobs to complete
for i in "${!pids[@]}"; do
  set +e
  wait "${pids[$i]}"
  code=$?
  set -e
  if [[ $code -eq 0 ]]; then
    echo "$(ts) ${names[$i]} finished OK"
  else
    echo "$(ts) ${names[$i]} failed with exit code $code"
  fi
done

echo "$(ts) All jobs done. Check logs/Qwen4b_rawModel_test_*.out and logs/Llama31_rawModel_test_*.out"
