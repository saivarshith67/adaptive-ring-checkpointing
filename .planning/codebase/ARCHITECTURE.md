# Architecture

**Analysis Date:** 2026-03-16

## Pattern Overview

**Overall:** Modular Machine Learning Training Pipeline with Strategy Pattern

**Key Characteristics:**
- Single-entry training script orchestrating all components
- Strategy pattern for checkpoint timing algorithms
- Factory pattern for strategy creation
- PyTorch-based deep learning model training
- Fault injection for testing checkpoint reliability

## Layers

**Configuration Layer:**
- Purpose: Load and validate runtime configuration from YAML
- Location: `src/coci/config.py`
- Contains: `Config` dataclass, `load_config()` function
- Depends on: `yaml` module
- Used by: `scripts/train.py`

**Model Layer:**
- Purpose: Factory for creating neural network models
- Location: `src/coci/models/model.py`
- Contains: `get_model()` function
- Depends on: `torch`, `torchvision.models`
- Used by: `scripts/train.py`

**Data Layer:**
- Purpose: Dataset loading and preprocessing
- Location: `src/coci/data_ingestor/`
- Contains: `get_cifar100_dataset()` in `cifar.py`, `ImageDataset` class in `dataset.py`
- Depends on: `torchvision`, `torch.utils.data`, `cv2`
- Used by: `scripts/train.py`

**Checkpointing Layer:**
- Purpose: Model state persistence and checkpoint timing strategies
- Location: `src/coci/checkpointing/`
- Contains: `CheckpointManager` class in `checkpoint_manager.py`, strategy classes in `strategy.py`
- Depends on: `torch`, `os`, `time`, `math`
- Used by: `scripts/train.py`

**Fault Injection Layer:**
- Purpose: Simulate runtime failures to test checkpoint/recovery
- Location: `src/coci/fault/fault_injector.py`
- Contains: `FaultInjector` class
- Depends on: `random`, `time`, `math`
- Used by: `scripts/train.py`

## Data Flow

**Training Flow:**

1. **Initialization**: Parse args → Load config → Create device → Initialize DataLoader → Create model → Setup optimizer → Initialize CheckpointManager → Create strategy
2. **Resume**: CheckpointManager.load_latest() restores model/optimizer state from disk
3. **Epoch Loop**: For each epoch, iterate through train_loader batches
4. **Forward/Backward**: Forward pass → Compute loss → Backward pass → Optimizer step
5. **Checkpoint Decision**: Strategy.should_checkpoint() called per batch → if true, CheckpointManager.save()
6. **Evaluation**: After each epoch, evaluate on test_loader
7. **Logging**: Write experiment summary to JSONL file

**Fault Injection Flow:**

1. FaultInjector initialized with failure_rate_per_second (Poisson process)
2. After each training batch, fault_injector.maybe_fail() called
3. If random value < Poisson probability, RuntimeError raised
4. Error caught in train.py main(), crash summary logged, process exits

## Key Abstractions

**CheckpointStrategy (Abstract Base Class):**
- Purpose: Interface for checkpoint timing algorithms
- Location: `src/coci/checkpointing/strategy.py`
- Examples: `FixedIntervalStrategy`, `YoungDalyStrategy`, `EpochStrategy`
- Pattern: Strategy pattern with ABC

**CheckpointStrategyFactory:**
- Purpose: Create strategy instances based on config
- Location: `src/coci/checkpointing/strategy.py`
- Examples: Creates strategy based on `cfg.strategy` ("fixed", "young_daly", "epoch")
- Pattern: Factory pattern

**CheckpointManager:**
- Purpose: Handle save/load of PyTorch model checkpoints
- Location: `src/coci/checkpointing/checkpoint_manager.py`
- Examples: save(), load_latest()
- Pattern: Repository pattern for model state

## Entry Points

**Training Entry:**
- Location: `scripts/train.py`
- Triggers: `python scripts/train.py --mode dev|server [--inject_fault]`
- Responsibilities: Orchestrate entire training pipeline, handle errors, log results

**Development Entry:**
- Location: `src/main.py`
- Triggers: `python src/main.py`
- Responsibilities: Placeholder (prints hello message)

## Error Handling

**Strategy:**
- Try-catch in train.py main() for RuntimeError from fault injection
- Graceful recovery via checkpoint loading
- Crash summaries logged to `crash_experiment_log_{strategy}.jsonl`

**Checkpoint Errors:**
- No checkpoint found → start from epoch 0
- Invalid checkpoint → would raise torch.load error (not caught)

## Cross-Cutting Concerns

**Logging:** Print statements with f-strings, experiment results to JSONL files

**Validation:** Config loaded via yaml.safe_load(), no explicit validation

**Authentication:** Not applicable (local training)

---

*Architecture analysis: 2026-03-16*
