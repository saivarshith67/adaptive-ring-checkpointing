#!/bin/bash

# -----------------------------------------------
# FaceForensics++ Deepfake Detection Training
# -----------------------------------------------

# --- GPU Selection ---
echo "Detecting available GPUs..."
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "No GPU detected. Exiting."
    exit 1
fi

echo "Found $GPU_COUNT GPU(s)."

# --- Training Mode ---
# Set to 1 for single-GPU, or use all GPUs for multi-GPU via torchrun
USE_MULTI_GPU=false  # Change to true to enable torchrun multi-GPU

if [ "$USE_MULTI_GPU" = true ] && [ "$GPU_COUNT" -gt 1 ]; then
    echo "Multi-GPU mode: using $GPU_COUNT GPUs via torchrun."
    LAUNCHER="torchrun --nproc_per_node=$GPU_COUNT"
    unset CUDA_VISIBLE_DEVICES
else
    echo "Single-GPU mode: selecting least-used GPU..."
    GPU_ID=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
        awk '{print NR-1 " " $1}' | sort -k2 -n | head -1 | awk '{print $1}')
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo "Selected GPU: $GPU_ID"
    LAUNCHER="python"
fi

# --- Environment ---
source ~/.bashrc
source .venv/bin/activate

# --- Training Arguments ---
# Customize these as needed
DATASET_PATH="./data/faceforensics"   # Path to FaceForensics++ dataset
COMPRESSION="c23"                      # c23 (visually lossless) or c40 (compressed)
MODEL="efficientnet_b0"                # efficientnet_b0 | resnet18 | resnet50 | mobilenet
EPOCHS=20
BATCH_SIZE=32
LR=1e-4
WEIGHT_DECAY=1e-5
NUM_WORKERS=4
CHECKPOINT_DIR="./checkpoints"
CHECKPOINT_INTERVAL=5
SPLIT=0.8

# Optional flags (uncomment to enable):
# EXTRA_FLAGS="--resume"            # Resume from latest checkpoint
# EXTRA_FLAGS="--no-pretrained"     # Train from scratch
# EXTRA_FLAGS="--limit 1000"        # Dev mode with limited data
# EXTRA_FLAGS="--no-precrop"        # Skip face crop precomputation
EXTRA_FLAGS=""

# --- Run Loop ---
while true; do
    echo "--------------------------------------------------"
    echo "Starting FaceForensics++ training..."
    echo "  Mode:        $( [ "$USE_MULTI_GPU" = true ] && echo "Multi-GPU ($GPU_COUNT)" || echo "Single GPU ($GPU_ID)" )"
    echo "  Model:       $MODEL"
    echo "  Epochs:      $EPOCHS"
    echo "  Batch Size:  $BATCH_SIZE"
    echo "  Dataset:     $DATASET_PATH"
    echo "--------------------------------------------------"

    $LAUNCHER scripts/train_faceforensics.py \
        --dataset-path "$DATASET_PATH" \
        --compression "$COMPRESSION" \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --weight-decay "$WEIGHT_DECAY" \
        --num-workers "$NUM_WORKERS" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --checkpoint-interval "$CHECKPOINT_INTERVAL" \
        --split "$SPLIT" \
        $EXTRA_FLAGS

    EXIT_CODE=$?
    echo "Training exited with code $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training completed successfully."
        break
    fi

    echo "Training crashed or interrupted. Restarting in 5 seconds..."
    sleep 5
done