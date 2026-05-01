#!/bin/bash

export TRAINING_MODE="fsdp"
export TRAIN_SCRIPT="scripts/train_fsdp.py"
export MODE_LABEL="FairScale / FSDP Checkpointing"
export CHECKPOINT_SUFFIX="fsdp"

bash scripts/run_faceforensics_framework_backend.sh
