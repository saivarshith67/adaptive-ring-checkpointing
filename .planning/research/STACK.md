# Technology Stack

**Project:** Adaptive Ring Checkpointing - Multi-GPU & Deep Fake Detection
**Researched:** 2026-03-16
**Confidence:** HIGH

## Recommended Stack

### Core Framework
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| PyTorch | 2.7.0+ (stable) | Deep learning framework | Current stable (Feb 2026), requires Python 3.10+. Supports CUDA 12.6/12.8. PyTorch 2.10 is in preview/nightly - stick to 2.7.x for production. |
| torchvision | Compatible with PyTorch 2.7 | Image transforms, pretrained models | Required for ResNet, image preprocessing |

### Multi-GPU Training
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| PyTorch DDP | Built-in (torch.nn.parallel.DistributedDataParallel) | Multi-GPU training | **Use DDP, NOT DataParallel.** DDP uses multi-process (1 process per GPU), avoids GIL contention, has better gradient sync, and scales to multi-node. DataParallel is single-process, multi-threaded, bottlenecked by GIL. |

### Face Detection & Preprocessing
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| facenet-pytorch | 2.5.3+ | MTCNN face detection | Standard for face detection in PyTorch. GPU-accelerated. Includes InceptionResnetV1 for embeddings. Last release Apr 2023 but actively maintained. |
| opencv-python (opencv-python-headless) | 4.10+ | Image loading, face crop | Required for image I/O. Use headless for server environments. |

### Image Augmentation
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| albumentations | 1.4+ | Image augmentation | Industry standard. 2-3x faster than torchvision transforms. Supports bounding boxes, keypoints (for face landmarks). Better for face images than basic transforms. |

### Dataset & Data Loading
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| kagglehub | Latest | Download Kaggle datasets | Official Kaggle API. Handles DFD dataset download. |
| pillow | 10.x | Image loading fallback | Used by torchvision datasets |
| numpy | 1.26+ | Array operations | Required by most ML libraries |

### Additional Utilities
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| tqdm | 4.66+ | Progress bars | Standard for training loops |
| pyyaml | 6.x | Config file parsing | Existing project uses YAML configs |

## Installation

```bash
# Core PyTorch with CUDA 12.6 support (most common)
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# For CUDA 12.8 (newer GPUs)
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Multi-GPU & utilities
pip install pyyaml tqdm

# Face detection
pip install facenet-pytorch opencv-python-headless

# Image augmentation
pip install albumentations

# Kaggle dataset download
pip install kagglehub

# Existing project dependencies (verify versions)
pip install numpy pillow
```

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|------------|---------|
| Multi-GPU | DDP | DataParallel (nn.DataParallel) | DataParallel is single-process with GIL bottleneck. Poor scaling beyond 2 GPUs. DDP is production-standard. |
| Face detection | facenet-pytorch | dlib, mediapipe | facenet-pytorch is pure PyTorch, GPU-accelerated end-to-end. dlib is slow, mediapipe is Google-centric. |
| Augmentation | albumentations | torchvision.transforms | Albumentations 2-3x faster, better for face images with keypoint support |
| Dataset API | kagglehub | kaggle CLI | kagglehub is newer, better API, recommended by Kaggle |
| Multi-GPU launch | torchrun / spawn | mp.spawn | torchrun is PyTorch's recommended entry point for DDP. Handles worker setup automatically. |

## Not Recommended

| Library | Why Avoid |
|---------|-----------|
| DataParallel (nn.DataParallel) | Single-process, GIL-bound, poor scaling. Use DDP instead. |
| keras/tensorflow | Project uses PyTorch. Mixing frameworks adds complexity. |
| imgaug | Albumentations is faster and better maintained. |
| dlib | Slow, not optimized for PyTorch pipeline. |

## Sources

- **PyTorch DDP:** PyTorch official tutorials (2025), Lambda Labs distributed training guide (2024-2025)
- **PyTorch version:** pytorch.org stable (2.7.0 as of Feb 2026)
- **Face detection:** facenet-pytorch GitHub (timesler/facenet-pytorch), active community
- **Augmentation:** Albumentations official docs
- **DFD Dataset:** Kaggle DFD Entire Original Dataset page

## Version Compatibility Notes

```
Python: 3.10+ (PyTorch 2.7+ requirement)
CUDA: 12.6 or 12.8 (12.1 deprecated in PyTorch 2.6+)
torchvision: Must match PyTorch version
```

## Confidence Assessment

| Area | Confidence | Reason |
|------|------------|--------|
| PyTorch version | HIGH | Verified on pytorch.org (Feb 2026) |
| DDP vs DataParallel | HIGH | Well-documented official recommendation |
| Face detection | HIGH | facenet-pytorch is standard in PyTorch ecosystem |
| Augmentation | HIGH | Albumentations widely adopted |
| Dataset download | MEDIUM | kagglehub is newer, may have API changes |
