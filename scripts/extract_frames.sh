#!/bin/bash

# -----------------------------------------------
# FaceForensics++ Frame Extraction Script
# Pre-extracts frames from videos to .npy files
# for fastest training loading
# -----------------------------------------------

# --- Parse Arguments ---
DATASET_PATH=${1:-"./data/faceforensics"}  # Path to FaceForensics++ dataset
NUM_FRAMES=${2:-30}                        # Frames to extract per video

# --- Environment ---
source ~/.bashrc 2>/dev/null
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# --- Run Frame Extraction ---
echo ""
echo "=================================================="
echo "FaceForensics++ Frame Extraction"
echo "=================================================="
echo "  Dataset:       $DATASET_PATH"
echo "  Frames/Video: $NUM_FRAMES"
echo "=================================================="
echo ""

CMD="python scripts/train_faceforensics.py \
    --extract-frames \
    --dataset-path \"$DATASET_PATH\" \
    --num-frames \"$NUM_FRAMES\""

echo "Extracting frames..."
echo "Command: $CMD"
echo ""

eval $CMD
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Frame extraction completed successfully."
    echo ""
    echo "You can now run training with:"
    echo "  bash scripts/run_faceforensics.sh"
else
    echo ""
    echo "Frame extraction failed with code $EXIT_CODE."
fi
