# Phase 3: Dataset Integration - Research

**Researched:** 2026-03-29
**Domain:** Deep Learning Training Pipeline - FaceForensics++ Dataset Integration
**Confidence:** HIGH

## Summary

This phase integrates the FaceForensics++ dataset from Kaggle with MTCNN face detection preprocessing and EfficientNet-B0 for binary classification. The research covers dataset download via kagglehub, face detection pipeline using facenet-pytorch, image transforms for 224x224 input, checkpoint compatibility with the new model architecture, and multi-GPU training via existing DDP infrastructure.

**Primary recommendation:** Use `kagglehub.dataset_download()` for FaceForensics++ dataset (hungle3401/faceforensics), MTCNN for face detection with torchvision transforms for 224x224 ImageNet-normalized input, and modify EfficientNet-B0 classifier for binary real/fake classification. Leverage existing Phase 1/2 DDP infrastructure for multi-GPU training.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use FaceForensics++ dataset from Kaggle via kagglehub (hungle3401/faceforensics)
- Use MTCNN from facenet-pytorch for face detection preprocessing
- Use EfficientNet-B0 as base classifier (alternative: ResNet-50)
- Image size: 224x224 for model input
- Multi-GPU via existing DDP infrastructure from Phase 1/2

### Claude's Discretion
- Specific data augmentation strategies (albumentations vs torchvision)
- Learning rate and optimizer settings
- Batch size tuning for GPU memory
- Number of epochs for training
- Whether to use pretrained weights or train from scratch

### Deferred Ideas (OUT OF SCOPE)
- Video frame extraction optimization (currently image-based)
- Real-time inference pipeline
- Advanced augmentation strategies
- Model ensemble approaches
- Multi-manipulation classification (Deepfakes, Face2Face, FaceSwap, NeuralTextures)
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CKPT-04 | Verify checkpoint compatibility with DDP state_dict keys | Existing checkpoint_manager.py uses model.module.state_dict() - compatible with DDP |
| DATA-04 | Integrate kagglehub for downloading FaceForensics++ dataset | kagglehub.dataset_download() API verified for Kaggle datasets |
| DATA-05 | Create dataset loader for FaceForensics++ image dataset | Dataset class structure defined with real/fake labels |
| DATA-06 | Add face detection preprocessing using MTCNN from facenet-pytorch | MTCNN API and integration patterns documented |
| DATA-07 | Configure image transforms compatible with face detection output | 224x224 transforms with ImageNet normalization specified |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| kagglehub | >=0.3.0 | Download FaceForensics++ from Kaggle | Official Kaggle Python client |
| facenet-pytorch | >=2.5.0 | MTCNN face detection | Most popular PyTorch MTCNN implementation |
| torch | >=2.0 | Core ML framework | Already in project |
| torchvision | >=0.15 | Models, transforms | Already in project |

### Supporting (Multi-GPU from Phase 1/2)
| Library | Purpose | When to Use |
|---------|---------|-------------|
| torch.distributed | DDP process group | Multi-GPU training (Phase 1) |
| DistributedSampler | Data partitioning | Multi-GPU training (Phase 1) |
| checkpoint_manager.py | Checkpoint save/load | Phase 2 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| kagglehub | kaggle CLI, manual download | kagglehub is programmatic, simpler for pipelines |
| facenet-pytorch | dlib MTCNN, OpenCV Haar | facenet-pytorch is pure PyTorch, well-maintained |
| EfficientNet-B0 | ResNet-50 | EfficientNet is more parameter-efficient (5.3M vs 25.6M params) |

**Installation:**
```bash
pip install kagglehub facenet-pytorch
```

---

## Dataset: FaceForensics++

### Dataset Overview
FaceForensics++ is a large-scale video dataset for forgery detection in human faces. The Kaggle version (hungle3401/faceforensics) contains:
- Original video sequences from YouTube (1000 videos)
- Manipulated sequences using Deepfakes, Face2Face, FaceSwap, and NeuralTextures
- Multiple compression levels (c23 = visually lossless, ~10GB)

### Expected Directory Structure (Kaggle version)
```
faceforensics/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           └── images/          # Real images (frames from original videos)
│               └── 000/         # Video folders
│               └── 001/
│               └── ...
├── manipulated_sequences/
│   └── Deepfakes/
│       └── c23/
│           └── images/          # Fake images (Deepfakes manipulation)
│               └── 000/
│               └── 001/
│               └── ...
│   └── Face2Face/
│   └── FaceSwap/
│   └── NeuralTextures/
```

### Binary Classification Labels
- **Real (0):** Images from `original_sequences/youtube/c23/images`
- **Fake (1):** Images from `manipulated_sequences/Deepfakes/c23/images` (using Deepfakes as primary manipulation method)

---

## Architecture Patterns

### Recommended Project Structure
```
src/
├── coci/
│   ├── data_ingestor/
│   │   ├── faceforensics.py    # NEW: FaceForensics++ dataset loader
│   │   ├── cifar.py            # Existing
│   │   └── dataset.py          # Existing base class
│   ├── models/
│   │   ├── model.py            # Modified for EfficientNet-B0
│   │   └── face_detector.py    # Existing/updated MTCNN wrapper
│   └── checkpointing/
│       └── checkpoint_manager.py  # Existing from Phase 2
│   └── distributed.py          # Existing from Phase 1
```

### Pattern 1: FaceForensics++ Dataset Loader with Face Detection
**What:** Dataset class that loads FaceForensics++ images, applies MTCNN face detection, and returns face-cropped images
**When to use:** For training binary classifier on FaceForensics++ dataset
**Example:**
```python
# Source: Research - FaceForensics++ dataset loader pattern
import os
from torch.utils.data import Dataset
from PIL import Image
from facenet_pytorch import MTCNN
import torchvision.transforms as transforms
import torch


class FaceForensicsDataset(Dataset):
    """FaceForensics++ dataset with MTCNN face detection preprocessing."""
    
    def __init__(self, root, split='train', transform=None, limit=None):
        """
        Args:
            root: Path to FaceForensics++ dataset root
            split: 'train', 'val', or 'test' for data split
            transform: Optional transforms for face-cropped images
            limit: Limit number of samples (for development)
        """
        # MTCNN face detector
        self.mtcnn = MTCNN(
            image_size=160,  # MTCNN native size
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.transform = transform or self._default_transform()
        
        # Load image paths with labels
        # Structure: root/original_sequences/youtube/c23/images/{video_id}/*.png
        #            root/manipulated_sequences/Deepfakes/c23/images/{video_id}/*.png
        images = []
        
        # Real images (label=0)
        real_root = os.path.join(root, 'original_sequences', 'youtube', 'c23', 'images')
        if os.path.exists(real_root):
            for video_folder in os.listdir(real_root):
                video_path = os.path.join(real_root, video_folder)
                if os.path.isdir(video_path):
                    for file in os.listdir(video_path):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            images.append((os.path.join(video_path, file), 0))  # 0 = real
        
        # Fake images (label=1) - using Deepfakes as primary manipulation
        fake_root = os.path.join(root, 'manipulated_sequences', 'Deepfakes', 'c23', 'images')
        if os.path.exists(fake_root):
            for video_folder in os.listdir(fake_root):
                video_path = os.path.join(fake_root, video_folder)
                if os.path.isdir(video_path):
                    for file in os.listdir(video_path):
                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                            images.append((os.path.join(video_path, file), 1))  # 1 = fake
        
        if limit:
            images = images[:limit]
        
        self.images = images
    
    def _default_transform(self):
        """Default transforms: 224x224 with ImageNet normalization."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        path, label = self.images[idx]
        
        # Load image
        img = Image.open(path).convert('RGB')
        
        # Detect and crop face using MTCNN
        try:
            # MTCNN returns cropped face tensor if face detected
            face = self.mtcnn(img)
            
            if face is None:
                # Fallback: resize original image if no face detected
                face = transforms.Resize((224, 224))(img)
            else:
                # MTCNN returns tensor [C, H, W], convert back to PIL for transforms
                face = transforms.ToPILImage()(face)
        except Exception as e:
            # Fallback on any error
            face = img
        
        # Apply transforms
        if self.transform:
            face = self.transform(face)
        
        return face, label
```

### Pattern 2: EfficientNet-B0 Binary Classifier
**What:** Load pretrained EfficientNet-B0 and modify for binary classification
**When to Use:** For deep fake detection binary classification
**Example:**
```python
# Source: torchvision.models.efficientnet_b0 documentation
import torchvision.models as models
import torch.nn as nn


def get_efficientnet_binary(pretrained=True):
    """
    Load EfficientNet-B0 modified for binary classification.
    
    Returns model with 2-class output for real/fake detection.
    """
    if pretrained:
        model = models.efficientnet_b0(weights='DEFAULT')
    else:
        model = models.efficientnet_b0(weights=None)
    
    # Get the number of input features to the classifier
    num_features = model.classifier[1].in_features
    
    # Replace classifier for binary classification
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, 2)
    )
    
    return model
```

### Pattern 3: kagglehub Dataset Download
**What:** Download FaceForensics++ dataset programmatically
**When to Use:** For automated dataset acquisition in training scripts
**Example:**
```python
# Source: kagglehub documentation for Kaggle datasets
import kagglehub
import os


def download_faceforensics_dataset():
    """
    Download FaceForensics++ dataset from Kaggle.
    
    Returns path to downloaded dataset.
    """
    # Download latest version of the dataset
    path = kagglehub.dataset_download('hungle3401/faceforensics')
    
    print(f"Dataset downloaded to: {path}")
    return path


# Usage
if __name__ == "__main__":
    dataset_path = download_faceforensics_dataset()
    # Expected structure: dataset_path/original_sequences/..., dataset_path/manipulated_sequences/...
```

### Pattern 4: Multi-GPU Training Integration
**What:** Update training_distributed.py to use FaceForensics++ dataset with DDP
**When to Use:** For distributed training with FaceForensics++ dataset
**Example:**
```python
# Source: Integration of Phase 1/2 DDP infrastructure with Phase 3 dataset
from src.coci.data_ingestor.faceforensics import FaceForensicsDataset, download_faceforensics_dataset
from src.coci.models.model import get_efficientnet_binary
from src.coci.distributed import setup_distributed, cleanup_distributed, is_main_process, barrier
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_faceforensics_dataloader(root, batch_size, num_workers, rank, world_size, train=True):
    """Get FaceForensics++ DataLoader with DistributedSampler for multi-GPU."""
    dataset = FaceForensicsDataset(
        root=root,
        split='train' if train else 'val',
        transform=None  # Use default transforms
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
        drop_last=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    return dataloader, sampler


# In main():
# Download dataset (once, on rank 0)
if is_main_process():
    dataset_path = download_faceforensics_dataset()
barrier()  # Wait for download to complete

# For binary classification with 2 classes
model = get_efficientnet_binary(pretrained=True)
model.to(device)

# Convert BatchNorm to SyncBatchNorm
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

# Wrap with DDP
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# Binary classification uses CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Face detection | Custom CNN for face detection | MTCNN from facenet-pytorch | Well-tested, pretrained, efficient |
| Dataset download | Manual download + extraction | kagglehub.dataset_download() | Programmatic, reproducible |
| ImageNet pretrained weights | Train from scratch | ImageNet pretrained EfficientNet-B0 | Faster convergence, better generalization |
| Image normalization | Custom mean/std | ImageNet stats (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225) | Matches pretrained model expectations |
| Multi-GPU training | Custom parallel code | Existing DDP from Phase 1/2 | Tested, efficient, standard PyTorch |
| Data partitioning | Custom sharding | DistributedSampler | Handles edge cases, maintains reproducibility |

**Key insight:** MTCNN handles face detection, alignment, and cropping in one pipeline. Building from scratch would require training a face detector model and implementing landmark detection for alignment. Use existing Phase 1/2 infrastructure for multi-GPU - it's already tested and working.

---

## Common Pitfalls

### Pitfall 1: MTCNN Device Management
**What goes wrong:** MTCNN defaults to CPU, causing slow training data loading
**Why it happens:** Device not specified during MTCNN initialization
**How to avoid:** Explicitly pass device during MTCNN creation
```python
# WRONG: Defaults to CPU
mtcnn = MTCNN(image_size=160)

# CORRECT: Use CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(image_size=160, device=device)
```

### Pitfall 2: Face Detection Failure Handling
**What goes wrong:** Dataset returns None when no face detected, causing training crashes
**Why it happens:** MTCNN returns None for images without detectable faces
**How to avoid:** Implement fallback in dataset's __getitem__
```python
# Handle no-face-detected case
face = self.mtcnn(img)
if face is None:
    # Fallback: use center crop or resize original
    face = transforms.CenterCrop(224)(img)
```

### Pitfall 3: Checkpoint Architecture Mismatch
**What goes wrong:** Loading checkpoint fails when num_classes changes between models
**Why it happens:** Different classifier layer dimensions
**How to avoid:** Save model config in checkpoint
```python
# In checkpoint_manager.py
torch.save({
    'epoch': epoch,
    'model_state_dict': state_dict,
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
    'model_config': {'num_classes': 2, 'model_name': 'efficientnet_b0'},
}, path)
```

### Pitfall 4: kagglehub Authentication
**What goes wrong:** kagglehub fails without Kaggle credentials
**Why it happens:** Missing kaggle.json credentials file
**How to avoid:** Set up credentials or use Kaggle API token
```python
# Option 1: Credentials file (default location)
# Windows: C:\Users\<username>\.kaggle\kaggle.json
# Linux/Mac: ~/.kaggle/kaggle.json

# Option 2: Environment variables
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_key'
```

### Pitfall 5: FaceForensics++ Dataset Structure
**What goes wrong:** Dataset path doesn't contain expected directory structure
**Why it happens:** FaceForensics++ structure differs from assumed format (video folders, compression levels)
**How to avoid:** Verify dataset structure after download, adapt paths as needed
```python
# Check what's in the downloaded dataset
import os
dataset_path = kagglehub.dataset_download('hungle3401/faceforensics')
print(os.listdir(dataset_path))  # Verify top-level structure
```

### Pitfall 6: Video vs Image Structure
**What goes wrong:** Kaggle dataset contains videos, not extracted frames
**Why it happens:** Some FaceForensics++ versions have videos that need frame extraction
**How to avoid:** Check if c23/images folder exists; if not, extract frames from videos using ffmpeg
```python
# If images don't exist, extract frames
# Use ffmpeg: ffmpeg -i video.mp4 -vf fps=1 frame_%04d.png
```

---

## Multi-GPU Training (Phase 1/2 Integration)

### Existing Infrastructure
Phase 1 and Phase 2 already provide:
- `src/coci/distributed.py`: Process group setup, rank detection, barrier
- `scripts/train_distributed.py`: torchrun-compatible entry point
- `src/coci/checkpointing/checkpoint_manager.py`: DDP-aware checkpoint save/load

### Integration Points
1. **Dataset loader** → returns batches compatible with existing training loop
2. **Model** → get_efficientnet_binary() returns model compatible with DDP wrapping
3. **Config** → DATASET_TYPE, MODEL_NAME, NUM_CLASSES updated for FaceForensics++

### Launch Command
```bash
# Multi-GPU training with torchrun
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    scripts/train_distributed.py \
    --dataset faceforensics \
    --model efficientnet_b0 \
    --batch_size 32
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Face crop manually | MTCNN automated detection | 2019+ (MTCNN release) | Reliable face detection |
| Train CNN from scratch | ImageNet pretrained + fine-tune | 2014+ (transfer learning普及) | Better convergence |
| Single GPU training | DDP multi-GPU | Phase 1/2 completed | Scale to multiple GPUs |
| CIFAR-100 | FaceForensics++ binary classification | Current phase | Domain-specific deepfake detection |

**Deprecated/outdated:**
- FaceNet InceptionResnetV1 for classification: Use EfficientNet-B0 instead (simpler, better documented)
- Custom face detection: MTCNN from facenet-pytorch is the standard
- DataParallel: Use DDP (Phase 1/2 infrastructure)

---

## Open Questions

1. **FaceForensics++ Dataset Structure on Kaggle**
   - What we know: Kaggle dataset slug is hungle3401/faceforensics
   - What's unclear: Exact folder structure in Kaggle version (images vs videos, compression levels)
   - Recommendation: After first download, inspect and adapt dataset loader

2. **Face Detection Performance**
   - What we know: MTCNN works well on frontal faces
   - What's unclear: Detection rate on FaceForensics++ (could contain various poses, quality)
   - Recommendation: Log detection failure rate during initial training runs

3. **Frame Extraction vs Pre-extracted Images**
   - What we know: FaceForensics++ is originally a video dataset
   - What's unclear: Does Kaggle version have extracted frames or videos?
   - Recommendation: Check for images folder; if not found, use ffmpeg for extraction

4. **MTCNN vs RetinaFace for Better Detection**
   - What we know: MTCNN is standard, RetinaFace may be more robust
   - What's unclear: Whether MTCNN accuracy is sufficient for FaceForensics++
   - Recommendation: Start with MTCNN, upgrade if needed

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (if tests exist) |
| Config file | none — existing project has no test infrastructure |
| Quick run command | N/A - no tests detected |
| Full suite command | N/A - no tests detected |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| CKPT-04 | Checkpoint compatibility with DDP state_dict | manual | Verify checkpoint save/load works with EfficientNet | ❌ |
| DATA-04 | kagglehub download | manual | `python -c "import kagglehub; print(kagglehub.dataset_download('hungle3401/faceforensics'))"` | ❌ |
| DATA-05 | Dataset loader | manual | Verify dataset returns correct batch shapes | ❌ |
| DATA-06 | MTCNN face detection | unit | Test face detection on sample images | ❌ |
| DATA-07 | Image transforms | unit | Verify 224x224 output with correct normalization | ❌ |

### Sampling Rate
- **Per task commit:** N/A - no automated tests
- **Per wave merge:** N/A
- **Phase gate:** Manual verification of all requirements

### Wave 0 Gaps
- [ ] `tests/test_faceforensics_dataset.py` — covers DATA-05, DATA-07
- [ ] `tests/test_mtcnn_detection.py` — covers DATA-06  
- [ ] `tests/test_checkpoint_ddp.py` — covers CKPT-04
- [ ] Framework install: `pip install pytest` — if tests desired
- [ ] `tests/conftest.py` — shared fixtures for dataset paths

*(Note: Project currently has no test infrastructure. All validation is manual.)*

---

## Sources

### Primary (HIGH confidence)
- kagglehub GitHub: https://github.com/Kaggle/kagglehub - Dataset download API
- facenet-pytorch GitHub: https://github.com/timesler/facenet-pytorch - MTCNN usage
- torchvision models documentation: https://docs.pytorch.org/vision/0.15/models/generated/torchvision.models.efficientnet_b0.html - EfficientNet-B0
- FaceForensics++ GitHub: https://github.com/ondyari/FaceForensics - Original dataset structure

### Secondary (MEDIUM confidence)
- EfficientNet transfer learning tutorials
- MTCNN integration patterns from facenet-pytorch examples
- PyTorch DDP documentation for multi-GPU integration

### Tertiary (LOW confidence)
- Kaggle FaceForensics++ dataset page: https://www.kaggle.com/datasets/hungle3401/faceforensics - needs verification after first download

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified, well-documented
- Architecture: HIGH - Existing DDP code, new dataset integration pattern clear
- Pitfalls: MEDIUM - Common issues identified, device/face detection edge cases
- Multi-GPU: HIGH - Phase 1/2 infrastructure already tested

**Research date:** 2026-03-29
**Valid until:** 2026-04-28 (30 days for stable stack)
