#!/bin/bash

echo "Detecting least-used GPU..."

GPU_ID=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
awk '{print NR-1 " " $1}' | sort -k2 -n | head -1 | awk '{print $1}')

if [ -z "$GPU_ID" ]; then
    echo "No GPU found. Exiting."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Selected GPU: $GPU_ID"

source ~/.bashrc
source .venv/bin/activate

# 🔥 Infinite restart loop
while true
do
    echo "Starting training..."
    python -m scripts.train --mode server --inject_fault

    EXIT_CODE=$?

    echo "Training exited with code $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully."
        break
    fi

    echo "Restarting in 5 seconds..."
    sleep 5
done