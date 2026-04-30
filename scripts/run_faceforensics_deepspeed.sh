#!/bin/bash

export TRAINING_MODE="deepspeed"
export TRAIN_SCRIPT="scripts/train_deepspeed.py"
export MODE_LABEL="DeepSpeed Checkpointing"
export CHECKPOINT_SUFFIX="deepspeed"

bash scripts/run_faceforensics_framework_backend.sh
