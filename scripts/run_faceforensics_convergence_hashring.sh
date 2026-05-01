#!/bin/bash

# -----------------------------------------------
# FaceForensics++ Training - Mode 4
# Convergence-aware timing + hash ring storage
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
CHECKPOINT_DIR="./checkpoints/convergence_hashring"
CHECKPOINT_INTERVAL=5
SPLIT=0.8

FAILURE_RATE_LAMBDA=0.00208
CHECKPOINT_COST_SEC=0.60
FIT_INTERVAL_STEPS=100
MIN_CONVERGENCE_INTERVAL_SEC=5.0
MAX_CONVERGENCE_INTERVAL_SEC=1800.0
AUTO_CALIBRATE_CHECKPOINT_COST=true
CHECKPOINT_COST_WARMUP_SAVES=3
CHECKPOINT_COST_EMA_ALPHA=0.5

HASH_RING_VIRTUAL_NODES=100
HASH_RING_CACHE_DIR="./checkpoints/hash_cache_convergence"

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
FAULT_LOG_PATH="$CHECKPOINT_DIR/fault_injection_convergence_hashring.jsonl"
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
echo "Mode: Convergence + Hash Ring"
echo "=================================================="
echo "  Profile:       $EXPERIMENT_PROFILE"
echo "  GPUs:          $NUM_GPUS"
echo "  Model:         $MODEL"
echo "  Epochs:        $EPOCHS"
echo "  Batch Size:    $BATCH_SIZE (per GPU)"
echo "  Dataset:       $DATASET_PATH"
echo "  Compression:   $COMPRESSION"
echo "  Lambda:        $FAILURE_RATE_LAMBDA"
echo "  CkptCost(sec): $CHECKPOINT_COST_SEC"
echo "  CkptCost Auto: $AUTO_CALIBRATE_CHECKPOINT_COST"
echo "  Ring VNodes:   $HASH_RING_VIRTUAL_NODES"
echo "  Ring Cache:    $HASH_RING_CACHE_DIR"
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

CMD="$LAUNCHER scripts/train_convergence_hashring.py \
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
    --failure-rate-lambda \"$FAILURE_RATE_LAMBDA\" \
    --checkpoint-cost-sec \"$CHECKPOINT_COST_SEC\" \
    --checkpoint-cost-warmup-saves \"$CHECKPOINT_COST_WARMUP_SAVES\" \
    --checkpoint-cost-ema-alpha \"$CHECKPOINT_COST_EMA_ALPHA\" \
    --fit-interval-steps \"$FIT_INTERVAL_STEPS\" \
    --min-convergence-interval-sec \"$MIN_CONVERGENCE_INTERVAL_SEC\" \
    --max-convergence-interval-sec \"$MAX_CONVERGENCE_INTERVAL_SEC\" \
    --hash-ring-virtual-nodes \"$HASH_RING_VIRTUAL_NODES\" \
    --hash-ring-cache-dir \"$HASH_RING_CACHE_DIR\""

if [ "$AUTO_CALIBRATE_CHECKPOINT_COST" = false ]; then
    CMD="$CMD --no-auto-calibrate-checkpoint-cost"
fi

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
