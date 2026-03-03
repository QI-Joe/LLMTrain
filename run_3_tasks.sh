#!/bin/bash

# Configuration
LOG_DIR="logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date '%m-%d_%H-%M')

# Function to start a background process and track its PID
pids=()
names=()

start() {
    local name="$1"
    shift
    echo "Starting $name..."
    # Execute the command in the background, redirecting output to log file
    "$@" > "${LOG_DIR}/${name}_${TIMESTAMP}.log" 2>&1 &
    local pid=$!
    pids+=($pid)
    names+=("$name")
    echo "$name started with PID $pid"
}

echo "Starting Multi-Task Training Run at $TIMESTAMP"

# --- Task 1: Llama 3.3 FSL (32 Shots) ---
# Assuming Llama 3.3 fits on GPU 0

# start "task1_llama3_SSL" \
# env CUDA_VISIBLE_DEVICES=0 python3 ZGeneration/train_gen.py \
#     --model_name "Llama-3.3-8B-Instruct" \
#     --cuda_device 0 \
#     --topic_name "Llama3_SSL" \
#     --experiment_name "Multi_Run_${TIMESTAMP}" \
#     --batch_size 2 \
#     --gradient_accumulation_steps 2 \
#     --num_epochs 5 \
#     --learning_rate 2e-5 \
#     --max_seq_length 2048 \
#     --max_new_tokens 50 \
#     --val_ratio 0.04 \
#     --test_ratio 0.04 \
#     --few_shot \
#     --shots_per_class 4 \
    # --semi_supervised \
    # --semi_ratio 0.01 \

# --- Task 2: Qwen 4B FSL (32 Shots) ---
# Run on GPU 1 (assuming available)
start "task2_qwen4b_FSL32" \
env CUDA_VISIBLE_DEVICES=1 python3 ZGeneration/train_gen.py \
    --model_name "Qwen3-4B-Instruct-2507" \
    --cuda_device 1 \
    --topic_name "Qwen4B_FSL16" \
    --experiment_name "Multi_Run_${TIMESTAMP}" \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_epochs 3 \
    --learning_rate 2e-5 \
    --max_seq_length 2183 \
    --max_new_tokens 50 \
    --val_ratio 0.04 \
    --test_ratio 0.04 \
    --few_shot \
    --shots_per_class 4 \
    # --semi_supervised \
    # --semi_ratio 0.1 \

# --- Task 3: Qwen 4B Semi-Supervised (SSL) ---
# Concurrent with Task 2 on GPU 1
start "task3_Llama31_FSL16" \
env CUDA_VISIBLE_DEVICES=2 python3 ZGeneration/train_gen.py \
    --model_name "llama3.1-8B-Instruct" \
    --cuda_device 2 \
    --topic_name "Llama3.1_FSL16" \
    --experiment_name "Multi_Run_${TIMESTAMP}" \
    --val_ratio 0.04 \
    --test_ratio 0.04 \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --num_epochs 5 \
    --learning_rate 2e-5 \
    --max_seq_length 2183 \
    --max_new_tokens 50 \
    --few_shot \
    --shots_per_class 4 \
    # --semi_supervised \
    # --semi_ratio 0.1 \

echo "All tasks submitted. Waiting for completion..."
echo "PIDs: ${pids[*]}"

# Monitor processes
# We loop and verify each process as it finishes.
# If any process exits with non-zero status (failure), we kill all remaining processes.

failed=0
count=${#pids[@]}

for ((i=0; i<count; i++)); do
    # wait -n waits for the next background job to finish and returns its exit status
    # Note: wait -n is available in bash 4.3+
    wait -n -p finished_pid
    exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "Process with PID $finished_pid failed with exit code $exit_code!"
        failed=1
        break
    else
        echo "Process with PID $finished_pid finished successfully."
    fi
done

if [ $failed -ne 0 ]; then
    echo "One or more tasks failed. Terminating remaining tasks..."
    for pid in "${pids[@]}"; do
        # Check if process is still running
        if ps -p $pid > /dev/null; then
            echo "Killing PID $pid"
            kill -TERM $pid 2>/dev/null
        fi
    done
    wait # Wait for kills to complete
    echo "Tasks terminated due to failure."
    exit 1
else
    echo "All tasks finished successfully."
fi
