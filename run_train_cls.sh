#!/usr/bin/env bash
# Run 3 Llama3 training tasks concurrently

set -euo pipefail
export PYTHONUNBUFFERED=1

mkdir -p logs

ts() { date '+%F %T'; }

pids=()
names=()

start() {
  local name="$1"; shift
  echo "$(ts) Starting $name ..."
  # Write stdout/stderr to logs/<name>.out
  "$@" >"logs/${name}.out" 2>&1 &
  pids+=($!)
  names+=("$name")
}
# ['Llama-3.3-8B-Instruct','Qwen3-4B-Instruct-2507', 'Qwen3-1.7B']
# [4096, 2560, 2048]

# Task 1: Basic Training (GPU 0)
start SSL1 \
python3 train_llama3.py \
  --model_name "Qwen3-1.7B" \
  --data_path "./data/emotion_cls" \
  --prompt_key "ws_prompt" \
  --dialogue_window 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --num_epochs 5 \
  --warmup_steps 500 \
  --gradient_accumulation_steps 1 \
  --topic_name "Qwen1d7_SSP_03" \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --num_workers 4 \
  --cuda_device 0 \
  --semi_supervised \
  --semi_ratio 0.5 \
  --log_interval 100 \
  --eval_interval 1000 \
  --save_interval 1000 \
  --output_dir "./outputs" \
  --seed 42 \
  --fast_train \
  --hidden_size 2048 \

# Task 2: Fast Training Mode (GPU 1)
start SSL2 \
python3 train_llama3.py \
  --model_name "Qwen3-4B-Instruct-2507" \
  --data_path "./data/emotion_cls" \
  --prompt_key "ws_prompt" \
  --dialogue_window 3 \
  --batch_size 6 \
  --learning_rate 2e-5 \
  --num_epochs 4 \
  --warmup_steps 500 \
  --gradient_accumulation_steps 2 \
  --topic_name "Qwen1d7_SSP_01" \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --num_workers 4 \
  --cuda_device 1 \
  --semi_supervised \
  --semi_ratio 0.4 \
  --log_interval 100 \
  --eval_interval 1000 \
  --save_interval 1000 \
  --output_dir "./outputs" \
  --seed 42 \
  --fast_train \
  --hidden_size 2560 \

# Task 3: Few-shot Learning (GPU 2)
start FSL1 \
python3 train_llama3.py \
  --model_name "Qwen3-4B-Instruct-2507" \
  --data_path "./data/emotion_cls" \
  --batch_size 6 \
  --learning_rate 2e-5 \
  --topic_name "Qwen4b_FSL32" \
  --num_epochs 4 \
  --cuda_device 2 \
  --few_shot \
  --shots_per_class 32 \
  --seed 42 \
  --prompt_key "ws_prompt" \
  --warmup_steps 100 \
  --gradient_accumulation_steps 2 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --num_workers 4 \
  --eval_interval 300 \
  --save_interval 300 \
  --fast_train \
  --hidden_size 2560 \
  # If you want fast mode here too, add: --fast_train

start FSL2 \
python3 train_llama3.py \
  --model_name "Llama-3.3-8B-Instruct" \
  --data_path "./data/emotion_cls" \
  --batch_size 4 \
  --learning_rate 2e-5 \
  --topic_name "Llama3.3_FSL24" \
  --num_epochs 5 \
  --cuda_device 3 \
  --few_shot \
  --shots_per_class 24 \
  --seed 42 \
  --prompt_key "ws_prompt" \
  --warmup_steps 100 \
  --gradient_accumulation_steps 2 \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --eval_interval 300 \
  --save_interval 300 \
  --num_workers 4 \
  --fast_train \
  --hidden_size 4096 \

# Cleanly stop children on Ctrl-C/TERM
trap 'echo; echo "Stopping all jobs..."; kill "${pids[@]}" 2>/dev/null || true' INT TERM

# Wait for all
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

echo "$(ts) All jobs done."