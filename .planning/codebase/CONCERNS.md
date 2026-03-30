# Codebase Concerns

**Analysis Date:** 2026-03-16

## Tech Debt

**Empty main.py entry point:**
- Issue: `src/main.py` only prints "Hello from adaptive-ring-checkpointing!" and does nothing
- Files: `src/main.py`
- Impact: No actual application entry point; all execution goes through `scripts/train.py`
- Fix approach: Implement proper main() that can invoke training or serve as CLI

**Config file field name mismatch:**
- Issue: `configs/dev.yaml` uses `failure_rate_per_sec` but `configs/server.yaml` uses `failure_rate_per_second`, while Config dataclass expects `failure_rate_per_second`
- Files: `configs/dev.yaml`, `src/coci/config.py`
- Impact: Loading dev.yaml will fail with missing field error
- Fix approach: Standardize field naming across config files

**Unused ImageDataset implementation:**
- Issue: `src/coci/data_ingestor/dataset.py` implements ImageDataset class but is never imported or used
- Files: `src/coci/data_ingestor/dataset.py`
- Impact: Dead code that will never run; training always uses CIFAR-100
- Fix approach: Either integrate ImageDataset or remove unused code

**EpochStrategy always returns True:**
- Issue: `EpochStrategy.should_checkpoint()` unconditionally returns `True`, checkpointing every batch
- Files: `src/coci/checkpointing/strategy.py` (lines 49-50)
- Impact: Excessive checkpointing, inefficient storage use, defeats purpose of adaptive strategy
- Fix approach: Implement proper epoch-boundary checkpoint logic

**No checkpoint load error handling:**
- Issue: `CheckpointManager.load_latest()` has no try/except around `torch.load()` - corrupted checkpoint crashes entire process
- Files: `src/coci/checkpointing/checkpoint_manager.py` (line 49)
- Impact: If checkpoint file is corrupted, entire resume fails with unhelpful error
- Fix approach: Wrap torch.load in try/except, offer recovery path

## Known Bugs

**cv2.imread returns None silently:**
- Issue: `ImageDataset.__getitem__()` calls `cv2.imread(path)` without checking if image loaded successfully
- Files: `src/coci/data_ingestor/dataset.py` (line 30)
- Trigger: Corrupted image file or invalid path
- Workaround: None - will cause downstream error

**No validation of config values:**
- Issue: `load_config()` directly passes YAML data to Config dataclass without validation
- Files: `src/coci/config.py` (line 20-24)
- Trigger: Invalid config values (negative epochs, zero batch size)
- Workaround: User must ensure valid config manually

**Checkpoint path not validated:**
- Issue: `CheckpointManager.save()` doesn't validate path existence before writing
- Files: `src/coci/checkpointing/checkpoint_manager.py`
- Impact: Silent failures if disk is full or permissions issues

## Security Considerations

**No secrets management:**
- Risk: No mechanism for API keys, tokens, or sensitive configuration
- Files: `src/coci/config.py`, config YAML files
- Current mitigation: None - configs checked into git
- Recommendations: Add environment variable support for sensitive values

**Checkpoint files not secured:**
- Risk: Checkpoint files contain model weights and optimizer state
- Files: `src/coci/checkpointing/checkpoint_manager.py`
- Current mitigation: None
- Recommendations: Consider encryption for checkpoints at rest

## Performance Bottlenecks

**Checkpoint strategy not optimal during resume:**
- Problem: After loading checkpoint, `strategy.update_checkpoint_time()` resets timer, causing immediate checkpoint
- Files: `scripts/train.py` (line 201), `src/coci/checkpointing/strategy.py`
- Cause: Fresh interval starts after resume regardless of actual elapsed time
- Improvement path: Track elapsed time properly across checkpoint/resume

**DataLoader num_workers=8 with cv2.imread:**
- Problem: Using OpenCV image loading in dataset with multiple workers can cause GIL contention
- Files: `scripts/train.py`, `src/coci/data_ingestor/dataset.py`
- Cause: cv2.imread is CPU-bound, workers compete for resources
- Improvement path: Use PIL or pre-load data

## Fragile Areas

**FixedIntervalStrategy missing super().__init__() call:**
- Files: `src/coci/checkpointing/strategy.py` (lines 17-27)
- Why fragile: Actually has proper super() call, but inconsistency in comments
- Safe modification: Already safe

**Strategy factory error handling:**
- Files: `src/coci/checkpointing/strategy.py` (lines 53-68)
- Why fragile: Only raises ValueError for unknown strategy, but doesn't handle None/missing config
- Safe modification: Add validation for required config fields

## Scaling Limits

**Single-threaded checkpoint operations:**
- Current capacity: One checkpoint at a time
- Limit: Large models take significant time to serialize
- Scaling path: Implement async checkpointing with background thread

**No distributed training support:**
- Current capacity: Single GPU
- Limit: Cannot scale to multi-GPU/multi-node
- Scaling path: Add DistributedDataParallel support

## Dependencies at Risk

**Loose version pinning in pyproject.toml:**
- Risk: Using `>=` for all dependencies can cause breaking changes
- Impact: Future installs may have incompatible API changes
- Migration plan: Pin exact versions or use tight bounds (e.g., `>=2.0.0,<3.0.0`)

**Heavy dependencies:**
- Risk: Full PyTorch + torchvision + CUDA requirements for a checkpointing demo
- Impact: Slow installs, large disk usage
- Migration plan: Consider lighter alternatives for non-ML components

## Missing Critical Features

**No actual ring checkpointing implementation:**
- Problem: Project name suggests "adaptive ring checkpointing" but no ring communication pattern found
- Blocks: Cannot actually do ring-based checkpoint aggregation

**No actual training loop fault recovery:**
- Problem: When RuntimeError is raised during fault injection, training just exits
- Blocks: True fault-tolerant training (checkpoint recovery and resume mid-epoch)

**Missing logging framework:**
- Problem: Using print() statements throughout
- Blocks: Proper log management, rotation, external log aggregation

## Test Coverage Gaps

**No test files at all:**
- What's not tested: All components
- Files: Entire codebase
- Risk: Any refactoring can break functionality unnoticed
- Priority: High

**No integration tests for checkpoint save/load cycle:**
- What's not tested: Checkpoint serialization/deserialization with actual model state
- Files: `src/coci/checkpointing/checkpoint_manager.py`
- Risk: Breaking changes in torch.save/load format or model architecture
- Priority: High

**No tests for fault injection:**
- What's not tested: FaultInjector behavior under various failure rates
- Files: `src/coci/fault/fault_injector.py`
- Risk: Incorrect failure rate calculation
- Priority: Medium

---

*Concerns audit: 2026-03-16*
