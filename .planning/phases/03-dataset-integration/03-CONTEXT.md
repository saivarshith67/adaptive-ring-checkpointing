# Phase 3: Dataset Integration - Context

**Gathered:** 2026-03-16
**Status:** Ready for planning
**Source:** User input - dataset change request

<domain>
## Phase Boundary

Integrate Deep Fake Detection (DFD) dataset with face detection preprocessing and optimize data pipeline.

This phase delivers:
- DFD dataset download and integration via kagglehub
- Dataset loader for face images
- MTCNN face detection preprocessing
- Image transforms for model input
- Checkpoint compatibility with new dataset
</domain>

<decisions>
## Implementation Decisions

### Dataset Change
- **Use Deep Fake Detection (DFD) dataset** from https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset
- Download via kagglehub library
- Dataset contains original face images for deep fake detection training

### Face Detection
- **Use MTCNN** from facenet-pytorch for face detection preprocessing
- Extract face bounding boxes from images
- Handle cases where no face is detected

### Model Architecture
- **Use EfficientNet-B0** as the base classifier (pretrained on ImageNet)
- Modify final layer for binary classification (real vs fake)
- Alternative: ResNet-50 if EfficientNet has issues

### Training Pipeline
- Update training script to use DFD dataset
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

- Dataset: DFD (Deep Fake Detection) - binary classification (real/fake)
- Face detection: MTCNN from facenet-pytorch
- Model: EfficientNet-B0 or ResNet-50
- Image transforms: 224x224, normalize with ImageNet stats
- Download: kagglehub library
</specifics>

<deferred>
## Deferred Ideas

- Video frame extraction (currently image-based dataset)
- Real-time inference pipeline
- Advanced augmentation strategies
- Model ensemble approaches
</deferred>

---

*Phase: 03-dataset-integration*
*Context gathered: 2026-03-16*
