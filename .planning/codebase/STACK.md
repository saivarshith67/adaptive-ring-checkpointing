# Technology Stack

**Analysis Date:** 2026-04-18

## Languages

**Primary:**
- Python 3.10.2 - All source code, training scripts, and core modules

## Runtime

**Environment:**
- Python (>=3.10.2)
- PyTorch 2.10.0

**Package Manager:**
- uv (via `uv pip compile pyproject.toml`)
- Lockfile: `uv.lock` present (405KB)
- Legacy: `requirements.txt` also maintained

## Frameworks

**Core:**
- PyTorch 2.10.0 - Deep learning framework
- TorchVision 0.25.0 - Computer vision utilities

**Data Processing:**
- OpenCV (cv2) 4.13.0.92 - Image/video processing
- NumPy 2.2.6 - Numerical computing
- Pandas 2.3.3 - Data manipulation
- Pillow 12.1.1 - Image handling

**Machine Learning:**
- scikit-learn 1.7.2 - ML utilities
- SciPy 1.15.3 - Scientific computing
- facenet-pytorch 2.5.3 - MTCNN face detection

**Visualization:**
- Matplotlib 3.10.8 - Plotting and visualization

**Other:**
- PyYAML 6.0.3 - Configuration file parsing
- tqdm 4.67.3 - Progress bars

## Key Dependencies

**Deep Learning:**
- `torch>=2.10.0` - PyTorch core
- `torchvision>=0.25.0` - Vision utilities
- `facenet-pytorch>=2.5.3` - MTCNN face detection

**Computer Vision:**
- `opencv-python>=4.13.0.92` - Image/video processing
- `Pillow>=12.1.1` - Image loading

**Data & Math:**
- `numpy>=2.2.6` - Arrays/matrices
- `pandas>=2.3.3` - DataFrames
- `scipy>=1.15.3` - Scientific computing
- `scikit-learn>=1.7.2` - ML algorithms

**Utilities:**
- `pyyaml>=6.0.3` - Config files
- `tqdm>=4.67.3` - Progress tracking
- `kagglehub>=0.3.0` - Dataset downloads (mentioned in pyproject.toml)

## Configuration

**Environment:**
- Python version: `.python-version` (3.10.2)
- Dependencies: `pyproject.toml` (source of truth), `requirements.txt` (generated)

**Build/Dev:**
- No explicit test framework configured
- No linting/formatting tools detected in project root
- `.ruff_cache/` directory suggests Ruff may be used

**Config Files:**
- `configs/dev.yaml` - Development settings (small dataset, 2 epochs, batch 8)
- `configs/server.yaml` - Production settings (full dataset, 20 epochs, batch 64)

## Platform Requirements

**Development:**
- Python 3.10.2+
- CUDA-capable GPU recommended for training
- ~2GB disk space for dependencies

**Production:**
- Multi-GPU distributed training supported via `torchrun`
- Requires CUDA for GPU acceleration
- Checkpoint-based fault tolerance for long training runs

---

*Stack analysis: 2026-04-18*