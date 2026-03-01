#!/bin/bash

echo "Detecting least-used GPU..."

# Get GPU with lowest memory usage
GPU_ID=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
awk '{print NR-1 " " $1}' | sort -k2 -n | head -1 | awk '{print $1}')

if [ -z "$GPU_ID" ]; then
    echo "Could not detect GPU. Exiting."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Selected GPU: $GPU_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Activate environment
source ~/.bashrc
source .venv/bin/activate

# Run training
python -m scripts.train --mode server --inject_fault