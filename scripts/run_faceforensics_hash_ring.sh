#!/bin/bash

# ------------------------------------------------------------
# FaceForensics++ Deepfake Detection Training (Hash-Ring Mode)
# ------------------------------------------------------------

# --- Parse Arguments ---
# Usage:
#   ./scripts/run_faceforensics_hash_ring.sh           # auto-detect GPUs
#   ./scripts/run_faceforensics_hash_ring.sh 4         # use 4 GPUs
#   ./scripts/run_faceforensics_hash_ring.sh 1         # single-GPU mode
NUM_GPUS_ARG=${1:-"auto"}

# --- GPU Selection ---
echo "Detecting available GPUs..."
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "No GPU detected. Exiting."
    exit 1
fi

echo "Found $GPU_COUNT GPU(s)."

# --- Determine number of GPUs to use ---
if [ "$NUM_GPUS_ARG" = "auto" ]; then
    # Keep behavior similar to the original script: cap at 4 to reduce contention.
    if [ "$GPU_COUNT" -ge 4 ]; then
        NUM_GPUS=4
    else
        NUM_GPUS=$GPU_COUNT
    fi
else
    NUM_GPUS=$NUM_GPUS_ARG
fi

if [ "$NUM_GPUS" -le 0 ]; then
    echo "Invalid GPU count: $NUM_GPUS"
    exit 1
fi

if [ "$NUM_GPUS" -gt "$GPU_COUNT" ]; then
    echo "Requested $NUM_GPUS GPU(s), but only $GPU_COUNT available."
    exit 1
fi

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
DATASET_PATH="./data/faceforensics/FF++"
COMPRESSION="c23"
MODEL="efficientnet_b0"
EPOCHS=20
BATCH_SIZE=2
LR=1e-4
WEIGHT_DECAY=1e-5
NUM_WORKERS=4
CHECKPOINT_DIR="./checkpoints"
CHECKPOINT_INTERVAL=5
SPLIT=0.8

# --- Hash Ring Checkpoint Settings ---
CHECKPOINT_MODE="hash-ring"
HASH_RING_VIRTUAL_NODES=100
HASH_RING_CACHE_DIR="./checkpoints/hash_cache"

# --- Video Mode Settings ---
USE_VIDEO_MODE=false
USE_FAST_VIDEO_MODE=true
NUM_FRAMES=16
TEMPORAL_MODEL="mean"

# --- Pre-extraction Settings ---
EXTRACT_FRAMES=false

# --- Optional Flags ---
EXTRA_FLAGS=""

# --- Run Training ---
echo ""
echo "=================================================="
echo "FaceForensics++ Deepfake Detection Training"
echo "=================================================="
echo "  GPUs:               $NUM_GPUS"
if [ "$USE_FAST_VIDEO_MODE" = true ]; then
    echo "  Mode:               Fast Video ($NUM_FRAMES frames, pre-extracted)"
elif [ "$USE_VIDEO_MODE" = true ]; then
    echo "  Mode:               Video ($NUM_FRAMES frames, $TEMPORAL_MODEL)"
else
    echo "  Mode:               Image"
fi
echo "  Model:              $MODEL"
echo "  Epochs:             $EPOCHS"
echo "  Batch Size:         $BATCH_SIZE (per GPU)"
echo "  Dataset:            $DATASET_PATH"
echo "  Compression:        $COMPRESSION"
echo "  Checkpoint Mode:    $CHECKPOINT_MODE"
echo "  Hash Ring VNodes:   $HASH_RING_VIRTUAL_NODES"
echo "  Hash Ring Cache:    $HASH_RING_CACHE_DIR"
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
    --split \"$SPLIT\" \
    --checkpoint-mode \"$CHECKPOINT_MODE\" \
    --hash-ring-virtual-nodes \"$HASH_RING_VIRTUAL_NODES\" \
    --hash-ring-cache-dir \"$HASH_RING_CACHE_DIR\""

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
