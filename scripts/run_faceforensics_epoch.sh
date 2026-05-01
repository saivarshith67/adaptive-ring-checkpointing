#!/bin/bash

# -----------------------------------------------
# FaceForensics++ Training - Mode 1
# Purely epoch based checkpointing
# -----------------------------------------------

EXPERIMENT_PROFILE=${EXPERIMENT_PROFILE:-baseline}
RESTART_DELAY_SEC=${RESTART_DELAY_SEC:-5}

echo "Detecting available GPUs..."
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "No GPU detected. Exiting."
    exit 1
fi

echo "Found $GPU_COUNT GPU(s)."

NUM_GPUS=4
if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
    NUM_GPUS=$GPU_COUNT
fi

echo "Using $NUM_GPUS GPU(s) for training."

# Function to select N least-used GPUs based on memory usage
select_least_used_gpus() {
    local N=$1
    # Query GPU memory usage, sort by memory used (ascending), pick top N
    # Output format: "gpu_id,gpu_id,..."
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | \
        sort -t',' -k2 -n | \
        head -n "$N" | \
        cut -d',' -f1 | \
        paste -sd ',' -
}

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Multi-GPU mode: selecting $NUM_GPUS least-used GPUs..."
    SELECTED_GPUS=$(select_least_used_gpus $NUM_GPUS)
    export CUDA_VISIBLE_DEVICES=$SELECTED_GPUS
    echo "Selected GPUs: $SELECTED_GPUS"
    LAUNCHER="torchrun --nproc_per_node=$NUM_GPUS"
else
    echo "Single-GPU mode: selecting least-used GPU..."
    GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
        sort -t',' -k2 -n | head -1 | cut -d',' -f1)
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo "Selected GPU: $GPU_ID"
    LAUNCHER="python"
fi

source ~/.bashrc 2>/dev/null
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

DATASET_PATH="./data/faceforensics/FF++"
COMPRESSION="c23"
MODEL="efficientnet_b0"
EPOCHS=20
BATCH_SIZE=2
LR=1e-4
WEIGHT_DECAY=1e-5
NUM_WORKERS=4
CHECKPOINT_DIR="./checkpoints/epoch"
CHECKPOINT_INTERVAL=5
SPLIT=0.8

USE_VIDEO_MODE=false
USE_FAST_VIDEO_MODE=true
NUM_FRAMES=16
TEMPORAL_MODEL="mean"

# Checkpoint fault injection settings (latest module in train_faceforensics.py)
# Note: injection is applied during checkpoint load, so resume must be enabled.
ENABLE_FAULT_INJECTION=false
AUTO_RESUME=false
ENABLE_RUNTIME_FAULT_INJECTION=false
RUNTIME_FAULT_AFTER_CHECKPOINTS=${RUNTIME_FAULT_AFTER_CHECKPOINTS:-1}
RUNTIME_FAULT_CRASH_AFTER_INJECTION=true
# Fault type selection (uncomment exactly one)
FAULT_TYPE="RANDOM_BIT"
# FAULT_TYPE="SPECIFIC_BIT"   # Requires FAULT_SPECIFIC_BIT to be set
# FAULT_TYPE="SIGN_BIT"
# FAULT_TYPE="EXPONENT_MSB"
FAULT_LOCATION="model"
FAULT_PROBABILITY=1.0
FAULT_BIT_FLIPS=1
FAULT_NUM_PROCESSES=1
FAULT_LOG_PATH="$CHECKPOINT_DIR/fault_injection_epoch.jsonl"
FAULT_BIT_RANGE=""
FAULT_TARGET_LAYERS=""
FAULT_SPECIFIC_BIT=""
FAULT_SEED=42

EXTRA_FLAGS=${EXTRA_FLAGS:-""}

if [ "$EXPERIMENT_PROFILE" = "resilience" ]; then
    AUTO_RESUME=true
    ENABLE_RUNTIME_FAULT_INJECTION=true
    RUNTIME_FAULT_CRASH_AFTER_INJECTION=true
elif [ "$EXPERIMENT_PROFILE" != "baseline" ]; then
    echo "Unsupported EXPERIMENT_PROFILE: $EXPERIMENT_PROFILE"
    echo "Use 'baseline' or 'resilience'."
    exit 1
fi

echo ""
echo "=================================================="
echo "FaceForensics++ Deepfake Detection Training"
echo "Mode: Pure Epoch Checkpointing"
echo "=================================================="
echo "  Profile:       $EXPERIMENT_PROFILE"
echo "  GPUs:          $NUM_GPUS"
echo "  Model:         $MODEL"
echo "  Epochs:        $EPOCHS"
echo "  Batch Size:    $BATCH_SIZE (per GPU)"
echo "  Dataset:       $DATASET_PATH"
echo "  Compression:   $COMPRESSION"
echo "  CheckpointDir: $CHECKPOINT_DIR"
echo "  Fault Inject:  $ENABLE_FAULT_INJECTION"
echo "  Runtime Fault: $ENABLE_RUNTIME_FAULT_INJECTION"
if [ "$ENABLE_FAULT_INJECTION" = true ] || [ "$ENABLE_RUNTIME_FAULT_INJECTION" = true ]; then
    echo "  Fault Type:    $FAULT_TYPE"
    echo "  Fault Loc:     $FAULT_LOCATION"
    echo "  Fault Prob:    $FAULT_PROBABILITY"
    echo "  Fault Flips:   $FAULT_BIT_FLIPS"
    echo "  Fault Procs:   $FAULT_NUM_PROCESSES"
fi
echo "=================================================="
echo ""

CMD="$LAUNCHER scripts/train_epoch_based.py \
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

if [ "$USE_FAST_VIDEO_MODE" = true ]; then
    CMD="$CMD --fast-video-mode --num-frames $NUM_FRAMES"
elif [ "$USE_VIDEO_MODE" = true ]; then
    CMD="$CMD --video-mode --num-frames $NUM_FRAMES --temporal-model $TEMPORAL_MODEL"
fi

if [ "$AUTO_RESUME" = true ]; then
    CMD="$CMD --resume"
fi

if [ "$ENABLE_RUNTIME_FAULT_INJECTION" = true ]; then
    CMD="$CMD --runtime-fault-injection --runtime-fault-after-checkpoints $RUNTIME_FAULT_AFTER_CHECKPOINTS"
    if [ "$RUNTIME_FAULT_CRASH_AFTER_INJECTION" = false ]; then
        CMD="$CMD --no-runtime-fault-crash-after-injection"
    fi
fi

if [ "$ENABLE_FAULT_INJECTION" = true ] || [ "$ENABLE_RUNTIME_FAULT_INJECTION" = true ]; then
    CMD="$CMD --fault-type $FAULT_TYPE --fault-location $FAULT_LOCATION --fault-probability $FAULT_PROBABILITY --fault-bit-flips $FAULT_BIT_FLIPS --fault-num-processes $FAULT_NUM_PROCESSES --fault-log-path \"$FAULT_LOG_PATH\" --fault-seed $FAULT_SEED"
    if [ -n "$FAULT_BIT_RANGE" ]; then
        CMD="$CMD --fault-bit-range $FAULT_BIT_RANGE"
    fi
    if [ -n "$FAULT_TARGET_LAYERS" ]; then
        CMD="$CMD --fault-target-layers $FAULT_TARGET_LAYERS"
    fi
    if [ -n "$FAULT_SPECIFIC_BIT" ]; then
        CMD="$CMD --fault-specific-bit $FAULT_SPECIFIC_BIT"
    fi
else
    CMD="$CMD --no-fault-injection"
fi

CMD="$CMD $EXTRA_FLAGS"

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
    echo "Training exited with code $EXIT_CODE. Restarting in $RESTART_DELAY_SEC seconds..."
    sleep "$RESTART_DELAY_SEC"
done
