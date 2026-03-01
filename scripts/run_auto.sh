#!/bin/bash

echo "Detecting least-used GPU..."

GPU_ID=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
awk '{print NR-1 " " $1}' | sort -k2 -n | head -1 | awk '{print $1}')

if [ -z "$GPU_ID" ]; then
    echo "No GPU detected. Exiting."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Selected GPU: $GPU_ID"

source ~/.bashrc
source .venv/bin/activate

while true
do
    echo "----------------------------------"
    echo "Starting training..."
    echo "----------------------------------"

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