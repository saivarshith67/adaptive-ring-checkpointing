#!/bin/bash

# -----------------------------------------------
# FaceForensics++ Deepfake Detection Training
# -----------------------------------------------

# --- Parse Arguments ---
# Allow overriding via command line
NUM_GPUS_ARG=${1:-"auto"}  # Default: auto-detect all GPUs

# --- GPU Selection ---
echo "Detecting available GPUs..."
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "No GPU detected. Exiting."
    exit 1
fi

echo "Found $GPU_COUNT GPU(s)."

# --- Determine number of GPUs to use ---
# Override: Use only 4 least-used GPUs to avoid memory contention
NUM_GPUS=4
echo "Using 4 least-used GPUs to avoid memory contention."

echo "Using $NUM_GPUS GPU(s) for training."

# --- Training Mode ---
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Multi-GPU mode: using $NUM_GPUS GPUs via torchrun."
    LAUNCHER="torchrun --nproc_per_node=$NUM_GPUS"
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
source ~/.bashrc 2>/dev/null
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# --- Training Arguments ---
DATASET_PATH="./data/faceforensics/FF++"   # Path to FaceForensics++ dataset
COMPRESSION="c23"                      # c23 (visually lossless) or c40 (compressed)
MODEL="efficientnet_b0"                # efficientnet_b0 | resnet18 | resnet50 | mobilenet
EPOCHS=20
BATCH_SIZE=2                           # Reduced for video mode (multiple frames)
LR=1e-4
WEIGHT_DECAY=1e-5
NUM_WORKERS=4
CHECKPOINT_DIR="./checkpoints"
CHECKPOINT_INTERVAL=5
SPLIT=0.8

# --- Video Mode Settings (from Kaggle notebook patterns) ---
# USE_VIDEO_MODE=true: Sample frames from video directories (slower but no preprocessing)
# USE_FAST_VIDEO_MODE=true: Use pre-extracted .npy frames (fastest loading)
USE_VIDEO_MODE=false                    # Video-level processing with frame sampling
USE_FAST_VIDEO_MODE=true                # Use pre-extracted .npy frames (recommended)
NUM_FRAMES=16                           # Frames per video (30 for fast mode from Kaggle)
TEMPORAL_MODEL="mean"                  # mean | lstm | gru | attention

# --- Pre-extraction Settings (for fast video mode) ---
EXTRACT_FRAMES=false                    # Set to true to extract frames before training

# --- Optional Flags ---
EXTRA_FLAGS=""

# --- Run Training ---
echo ""
echo "=================================================="
echo "FaceForensics++ Deepfake Detection Training"
echo "=================================================="
echo "  GPUs:          $NUM_GPUS"
if [ "$USE_FAST_VIDEO_MODE" = true ]; then
    echo "  Mode:          Fast Video ($NUM_FRAMES frames, pre-extracted)"
elif [ "$USE_VIDEO_MODE" = true ]; then
    echo "  Mode:          Video ($NUM_FRAMES frames, $TEMPORAL_MODEL)"
else
    echo "  Mode:          Image"
fi
echo "  Model:         $MODEL"
echo "  Epochs:        $EPOCHS"
echo "  Batch Size:    $BATCH_SIZE (per GPU)"
echo "  Dataset:       $DATASET_PATH"
echo "  Compression:   $COMPRESSION"
echo "=================================================="
echo ""

# Build command
CMD="$LAUNCHER scripts/train_faceforensics.py \
    --dataset-path \"$DATASET_PATH\" \
    --compression \"$COMPRESSION\" \
    --model \"$MODEL\" \
    --epochs \"$EPOCHS\" \
    --batch-size \"$BATCH_SIZE\" \
    --lr \"$LR\" \
    --weight-decay \"$WEIGHT_DECAY\" \
    --num-workers \"$NUM_WORKERS\" \
    --checkpoint-dir \"$CHECKPOINT_DIR\" \
    --checkpoint-interval \"$CHECKPOINT_INTERVAL\" \
    --split \"$SPLIT\""

# Add video mode flags
if [ "$USE_FAST_VIDEO_MODE" = true ]; then
    CMD="$CMD --fast-video-mode --num-frames $NUM_FRAMES"
elif [ "$USE_VIDEO_MODE" = true ]; then
    CMD="$CMD --video-mode --num-frames $NUM_FRAMES --temporal-model $TEMPORAL_MODEL"
fi

# Add extra flags
CMD="$CMD $EXTRA_FLAGS"

# Run with restart loop
while true; do
    echo "Starting training..."
    echo "Command: $CMD"
    echo ""

    eval $CMD
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "Training completed successfully."
        break
    fi

    echo ""
    echo "Training exited with code $EXIT_CODE. Restarting in 5 seconds..."
    sleep 5
done
