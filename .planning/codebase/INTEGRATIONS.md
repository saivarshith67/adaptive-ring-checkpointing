# External Integrations

**Analysis Date:** 2026-04-18

## APIs & External Services

**Dataset Downloads:**
- **Kaggle** - FaceForensics++ dataset download
  - SDK/Client: `kagglehub` (>=0.3.0)
  - Dataset: `hungle3401/faceforensics`
  - Usage: `src/coci/data_ingestor/faceforensics.py` - `download_faceforensics_dataset()`

**Model Checkpoints:**
- **facenet-pytorch** - Pre-trained MTCNN for face detection
  - Package: `facenet-pytorch>=2.5.3`
  - Purpose: Face detection in FaceForensics++ dataset
  - Model: MTCNN (Multi-task Cascaded Convolutional Networks)

**Model Weights:**
- **TorchVision** - Pre-trained model weights
  - Package: `torchvision>=0.25.0`
  - Used for: ResNet18, ResNet50 model initialization
  - Source: `torchvision.models`

## Data Storage

**Local Filesystem:**
- Checkpoint directory: `checkpoints/`
- Dataset directory: `data/`
- Config directory: `configs/`
- Logs: JSONL files in project root

**No Remote Storage:**
- No cloud storage integration (S3, GCS, etc.)
- No database connections
- All data stored locally

## Authentication & Identity

**No External Auth:**
- No authentication providers
- No OAuth/SSO integrations
- Training runs are local-only

## Monitoring & Observability

**No External Monitoring:**
- No error tracking service (Sentry, etc.)
- No logging service (Loggly, etc.)
- No metrics backend (Prometheus, etc.)

**Local Logging:**
- JSONL experiment logs in project root:
  - `experiment_log.jsonl`
  - `experiment_log_epoch.jsonl`
  - `experiment_log_young_daly.jsonl`
  - `experimental_log_*.jsonl`

## CI/CD & Deployment

**No CI/CD:**
- No CI pipeline configured
- No deployment automation
- No Docker/container support
- No cloud deployment configs

## Environment Configuration

**No Environment Variables Required:**
- No mandatory env vars for basic operation
- Optional: `CUDA_VISIBLE_DEVICES` for GPU selection

**torchrun Environment Variables (for distributed training):**
- `LOCAL_RANK` - Local GPU device index
- `RANK` - Global process rank
- `WORLD_SIZE` - Total number of processes

## Webhooks & Callbacks

**None:**
- No incoming webhooks
- No outgoing webhooks
- No callback services

---

*Integration audit: 2026-04-18*