#!/bin/bash

# Bind to GPU 1
export CUDA_VISIBLE_DEVICES=1

echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Activate environment
source ~/.bashrc
source .venv/bin/activate   # adjust if needed

# Run training
python -m scripts.train --mode server --inject_fault