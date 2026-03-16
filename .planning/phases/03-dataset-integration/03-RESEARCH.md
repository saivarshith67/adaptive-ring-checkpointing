# Phase 3: Dataset Integration - Research

**Researched:** 2026-03-16
**Domain:** Deep Learning Training Pipeline - Dataset Integration
**Confidence:** HIGH

## Summary

This phase integrates the Deep Fake Detection (DFD) dataset from Kaggle with MTCNN face detection preprocessing and EfficientNet-B0 for binary classification. The research covers dataset download via kagglehub, face detection pipeline using facenet-pytorch, image transforms for 224x224 input, and checkpoint compatibility with the new model architecture.

**Primary recommendation:** Use `kagglehub.dataset_download()` for DFD dataset, MTCNN for face detection with torchvision transforms for 224x224 ImageNet-normalized input, and modify EfficientNet-B0 classifier for binary real/fake classification.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Use Deep Fake Detection (DFD) dataset from Kaggle via kagglehub
- Use MTCNN from facenet-pytorch for face detection preprocessing
- Use EfficientNet-B0 as base classifier (alternative: ResNet-50)
- Image size: 224x224 for model input

### Claude's Discretion
- Specific data augmentation strategies (albumentations vs torchvision)
- Learning rate and optimizer settings
- Batch size tuning for GPU memory
- Number of epochs for training
- Whether to use pretrained weights or train from scratch

### Deferred Ideas (OUT OF SCOPE)
- Video frame extraction (currently image-based dataset)
- Real-time inference pipeline
- Advanced augmentation strategies
- Model ensemble approaches
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CKPT-04 | Verify checkpoint compatibility with DDP state_dict keys | Existing checkpoint_manager.py uses model.module.state_dict() - compatible with DDP |
| DATA-04 | Integrate kagglehub for downloading DFD dataset | kagglehub.dataset_download() API verified |
| DATA-05 | Create dataset loader for Deep Fake Detection image dataset | Dataset class structure defined with real/fake labels |
| DATA-06 | Add face detection preprocessing using MTCNN from facenet-pytorch | MTCNN API and integration patterns documented |
| DATA-07 | Configure image transforms compatible with face detection output | 224x224 transforms with ImageNet normalization specified |
</phase_requirements>

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| kagglehub | >=0.3.0 | Download DFD dataset from Kaggle | Official Kaggle Python client |
| facenet-pytorch | >=2.5.0 | MTCNN face detection | Most popular PyTorch MTCNN implementation |
| torch | >=2.0 | Core ML framework | Already in project |
| torchvision | >=0.15 | Models, transforms | Already in project |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Pillow | latest | Image loading for MTCNN | Required by facenet-pytorch |
| opencv-python | latest | Image loading in dataset | Already in project (cv2) |

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

## Architecture Patterns

### Recommended Project Structure
```
src/
├── coci/
│   ├── data_ingestor/
│   │   ├── dfd.py              # NEW: DFD dataset loader
│   │   ├── cifar.py            # Existing
│   │   └── dataset.py          # Existing base class
│   ├── models/
│   │   ├── model.py            # Modified for EfficientNet-B0
│   │   └── face_detector.py    # NEW: MTCNN wrapper
│   └── checkpointing/
│       └── checkpoint_manager.py  # Modified for model config
```

### Pattern 1: DFD Dataset Loader with Face Detection
**What:** Dataset class that loads DFD images, applies MTCNN face detection, and returns face-cropped images
**When to use:** For training binary classifier on DFD dataset
**Example:**
```python
# Source: Research - dataset loader pattern
import os
from torch.utils.data import Dataset
from PIL import Image
from facenet_pytorch import MTCNN
import torchvision.transforms as transforms


class DFDFaceDataset(Dataset):
    """DFD dataset with MTCNN face detection preprocessing."""
    
    def __init__(self, root, transform=None, limit=None):
        """
        Args:
            root: Path to DFD dataset (contains 'real' and 'fake' subdirs)
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
        images = []
        for label_name in ["real", "fake"]:
            folder = os.path.join(root, label_name)
            if os.path.exists(folder):
                for file in os.listdir(folder):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        images.append((os.path.join(folder, file), label_name))
        
        if limit:
            images = images[:limit]
        
        self.images = images
        self.label_map = {"real": 0, "fake": 1}
    
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
        
        label_idx = self.label_map[label]
        
        return face, label_idx
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
    # Option 1: 2-class output (for CrossEntropyLoss)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(num_features, 2)
    )
    
    return model
```

### Pattern 3: kagglehub Dataset Download
**What:** Download DFD dataset programmatically
**When to Use:** For automated dataset acquisition in training scripts
**Example:**
```python
# Source: kagglehub GitHub documentation
import kagglehub
import os


def download_dfd_dataset():
    """
    Download Deep Fake Detection dataset from Kaggle.
    
    Returns path to downloaded dataset.
    """
    # Download latest version of the dataset
    path = kagglehub.dataset_download('sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset')
    
    print(f"Dataset downloaded to: {path}")
    return path


# Usage
if __name__ == "__main__":
    dataset_path = download_dfd_dataset()
    # Expected structure: dataset_path/real/*, dataset_path/fake/*
```

### Pattern 4: Training Script Integration
**What:** Update training_distributed.py to use DFD dataset
**When to Use:** For distributed training with DFD dataset
**Example:**
```python
# Source: Modified train_distributed.py pattern
from src.coci.data_ingestor.dfd import DFDFaceDataset
from src.coci.models.model import get_efficientnet_binary


def get_dfd_dataset(root, train=True, transform=None):
    """Get DFD dataset with face detection."""
    dataset = DFDFaceDataset(
        root=root,
        transform=transform,
        limit=None  # Remove limit for full training
    )
    return dataset


# In main():
# Replace CIFAR with DFD
train_dataset = get_dfd_dataset(
    root="./data/dfd",  # After kagglehub download
    train=True
)
# For binary classification with 2 classes
model = get_efficientnet_binary(pretrained=True)
model.to(device)

# Convert BatchNorm to SyncBatchNorm
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

# Wrap with DDP
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# Binary classification uses CrossEntropyLoss or BCEWithLogitsLoss
criterion = nn.CrossEntropyLoss()  # 2-class output
# OR
criterion = nn.BCEWithLogitsLoss()  # 1 output with sigmoid
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Face detection | Custom CNN for face detection | MTCNN from facenet-pytorch | Well-tested, pretrained, efficient |
| Dataset download | Manual download + extraction | kagglehub.dataset_download() | Programmatic, reproducible |
| ImageNet pretrained weights | Train from scratch | ImageNet pretrained EfficientNet-B0 | Faster convergence, better generalization |
| Image normalization | Custom mean/std | ImageNet stats (0.485, 0.456, 0.406) / (0.229, 0.224, 0.225) | Matches pretrained model expectations |

**Key insight:** MTCNN handles face detection, alignment, and cropping in one pipeline. Building from scratch would require training a face detector model and implementing landmark detection for alignment.

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
    'model_config': {'num_classes': 2, 'model_name': 'efficientnet_b0'},  # NEW
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

### Pitfall 5: DFD Dataset Structure
**What goes wrong:** Dataset path doesn't contain expected real/fake subdirectories
**Why it happens:** DFD dataset structure differs from assumed format
**How to avoid:** Verify dataset structure after download
```python
# Check what's in the downloaded dataset
import os
dataset_path = kagglehub.dataset_download('sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset')
print(os.listdir(dataset_path))  # Verify structure
```

---

## Code Examples

### Image Transforms for EfficientNet-B0
```python
# Source: torchvision.models.EfficientNet_B0_Weights documentation
from torchvision import transforms

# Training transforms with augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Slightly larger for random crop
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Validation transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### DDP Checkpoint with Model Config
```python
# Source: checkpoint_manager.py modification for architecture change
def save(self, model, optimizer, epoch, loss, model_config=None):
    """Save checkpoint with optional model config."""
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank != 0:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        return
    
    path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    state_dict = (
        model.module.state_dict() if self.is_ddp_wrapped else model.state_dict()
    )
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    
    # Add model config if provided
    if model_config:
        checkpoint["model_config"] = model_config
    
    torch.save(checkpoint, path)
    
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def load_latest(self, model, optimizer, device):
    """Load checkpoint, optionally validate architecture."""
    # ... existing loading code ...
    
    # Validate model config if present
    if "model_config" in checkpoint:
        saved_config = checkpoint["model_config"]
        # Could validate num_classes matches, etc.
        print(f"Checkpoint model config: {saved_config}")
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Face crop manually | MTCNN automated detection | 2019+ (MTCNN release) | Reliable face detection |
| Train CNN from scratch | ImageNet pretrained + fine-tune | 2014+ (transfer learning普及) | Better convergence |
| Single GPU training | DDP multi-GPU | Standard in PyTorch | Scale to multiple GPUs |
| CIFAR-100 | DFD binary classification | Current phase | Domain-specific task |

**Deprecated/outdated:**
- FaceNet InceptionResnetV1 for classification: Use EfficientNet-B0 instead (simpler, better documented)
- Custom face detection: MTCNN from facenet-pytorch is the standard

---

## Open Questions

1. **DFD Dataset Structure Verification**
   - What we know: kagglehub downloads to a path, but exact subdirectory structure needs verification
   - What's unclear: Whether DFD has train/val/test split, exact folder names (case-sensitive?)
   - Recommendation: After first download, inspect and adapt dataset loader

2. **Face Detection Performance**
   - What we know: MTCNN works well on frontal faces
   - What's unclear: Detection rate on DFD dataset (could contain various poses)
   - Recommendation: Log detection failure rate during initial training runs

3. **MTCNN vs RetinaFace for Better Detection**
   - What we know: MTCNN is standard, RetinaFace may be more robust
   - What's unclear: Whether MTCNN accuracy is sufficient for DFD
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
| DATA-04 | kagglehub download | manual | `python -c "import kagglehub; print(kagglehub.dataset_download('sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset'))"` | ❌ |
| DATA-05 | Dataset loader | manual | Verify dataset returns correct batch shapes | ❌ |
| DATA-06 | MTCNN face detection | unit | Test face detection on sample images | ❌ |
| DATA-07 | Image transforms | unit | Verify 224x224 output with correct normalization | ❌ |

### Sampling Rate
- **Per task commit:** N/A - no automated tests
- **Per wave merge:** N/A
- **Phase gate:** Manual verification of all requirements

### Wave 0 Gaps
- [ ] `tests/test_dfd_dataset.py` — covers DATA-05, DATA-07
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

### Secondary (MEDIUM confidence)
- EfficientNet transfer learning tutorials on DebuggerCafe
- MTCNN integration patterns from facenet-pytorch examples

### Tertiary (LOW confidence)
- Web search for DFD dataset structure - needs verification after first download

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified, well-documented
- Architecture: HIGH - Existing DDP code, new dataset integration pattern clear
- Pitfalls: MEDIUM - Common issues identified, device/face detection edge cases

**Research date:** 2026-03-16
**Valid until:** 2026-04-16 (30 days for stable stack)
