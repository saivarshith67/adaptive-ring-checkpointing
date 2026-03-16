# External Integrations

**Analysis Date:** 2026-03-16

## APIs & External Services

**Not applicable** - This is a standalone deep learning training project without external API integrations.

## Data Storage

**Local Filesystem:**
- Checkpoint storage: `checkpoints/` directory
  - Format: PyTorch `.pt` files
  - Managed by: `src/coci/checkpointing/checkpoint_manager.py`
  - Implementation: Local file I/O using `torch.save()` and `torch.load()`

**Datasets:**
- CIFAR-100: Downloaded via torchvision
  - Location: `./data` directory (configurable)
  - Source: PyTorch official dataset repository
  - Auto-download: Enabled in `src/coci/data_ingestor/cifar.py`

- Custom Image Dataset: Local folder structure
  - Location: Configurable via `dataset_path` in YAML configs
  - Structure: `{root}/real/` and `{root}/fake/` subdirectories
  - Implementation: `src/coci/data_ingestor/dataset.py` using OpenCV

**Experiment Logs:**
- JSONL format for experiment results
- Output files: `final_experiment_log_{strategy}.jsonl`, `crash_experiment_log_{strategy}.jsonl`
- Implementation: Direct file writes in `scripts/train.py`

## Authentication & Identity

**Not applicable** - No authentication required. Training runs locally.

## Monitoring & Observability

**Logging:**
- Console output via `print()` statements
- No structured logging framework

**Metrics:**
- Custom metrics tracked in code:
  - Checkpoint count: `checkpoint_manager.num_checkpoints`
  - Total checkpoint time: `checkpoint_manager.total_checkpoint_time`
  - Failure statistics: `fault_injector.get_stats()`
  - Runtime: `total_runtime` calculated in `scripts/train.py`

**Error Tracking:**
- Not integrated with external error tracking services

## CI/CD & Deployment

**Hosting:**
- Not applicable - Local execution only

**CI Pipeline:**
- Not configured

## Environment Configuration

**Required env vars:**
- None detected - All configuration via YAML files

**Config file approach:**
- `configs/dev.yaml` - Development configuration
- `configs/server.yaml` - Production/Server configuration

**Configurable parameters:**
| Parameter | Description |
|-----------|-------------|
| dataset_path | Path to custom dataset |
| dataset_limit | Limit number of samples |
| batch_size | Training batch size |
| epochs | Number of training epochs |
| num_workers | DataLoader workers |
| model | Model architecture (resnet18, resnet50, mobilenet) |
| checkpoint_interval | Checkpoint frequency |
| failure_rate_per_second | Fault injection rate |
| strategy | Checkpoint strategy (fixed, young_daly, epoch) |
| fixed_interval | Fixed interval seconds |
| checkpoint_cost_estimate | Estimated checkpoint cost |

## Webhooks & Callbacks

**Incoming:**
- Not applicable - No HTTP endpoints

**Outgoing:**
- Not applicable - No external notifications or webhooks

---

*Integration audit: 2026-03-16*
