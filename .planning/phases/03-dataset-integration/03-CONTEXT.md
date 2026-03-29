# Phase 3: Dataset Integration - Context

**Gathered:** 2026-03-29
**Status:** Ready for planning
**Source:** User input - FaceForensics++ dataset specification

<domain>
## Phase Boundary

Integrate FaceForensics++ dataset with face detection preprocessing and optimize data pipeline for multi-GPU training.

This phase delivers:
- FaceForensics++ dataset download and integration via kagglehub
- Dataset loader for face images (original/manipulated)
- MTCNN face detection preprocessing
- Image transforms for model input
- Checkpoint compatibility with new dataset
- Multi-GPU training support (via existing DDP infrastructure)
</domain>

<decisions>
## Implementation Decisions

### Dataset Change (LOCKED)
- **Use FaceForensics++ dataset** from https://www.kaggle.com/datasets/hungle3401/faceforensics
- Download via kagglehub library
- Binary classification: "real" (original sequences) vs "fake" (manipulated sequences)
- Use c23 compression (visually lossless, ~10GB total)

### Dataset Structure
- **Real images:** original_sequences/youtube/c23/images
- **Fake images:** manipulated_sequences/Deepfakes/c23/images
- Frame extraction from videos may be needed

### Face Detection
- **Use MTCNN** from facenet-pytorch for face detection preprocessing
- Extract face bounding boxes from images
- Handle cases where no face is detected

### Model Architecture
- **Use EfficientNet-B0** as the base classifier (pretrained on ImageNet)
- Modify final layer for binary classification (real vs fake)
- Alternative: ResNet-50 if EfficientNet has issues

### Multi-GPU Training (LOCKED - existing Phase 1 infrastructure)
- Use existing distributed.py for process group setup
- Use torchrun for multi-GPU launch
- Use DistributedSampler for data partitioning
- NCCL backend for GPU communication

### Training Pipeline
- Keep existing DDP and checkpoint integration from Phase 2
- Face detection preprocessing in data pipeline
- Image size: 224x224 for model input

### Claude's Discretion
- Specific data augmentation strategies (albumentations vs torchvision)
- Learning rate and optimizer settings
- Batch size tuning for GPU memory
- Number of epochs for training
- Whether to use pretrained weights or train from scratch
</decisions>

<specifics>
## Specific Ideas

- Dataset: FaceForensics++ (hungle3401/faceforensics) - binary classification (real/fake)
- Face detection: MTCNN from facenet-pytorch
- Model: EfficientNet-B0 or ResNet-50
- Image transforms: 224x224, normalize with ImageNet stats
- Download: kagglehub library
- Multi-GPU: torchrun --nproc_per_node=N scripts/train_distributed.py
</specifics>

<deferred>
## Deferred Ideas

- Video frame extraction optimization (currently image-based)
- Real-time inference pipeline
- Advanced augmentation strategies
- Model ensemble approaches
- FaceForensics++ manipulation method classification (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
</deferred>

---

*Phase: 03-dataset-integration*
*Context gathered: 2026-03-29 via FaceForensics++ dataset specification*
