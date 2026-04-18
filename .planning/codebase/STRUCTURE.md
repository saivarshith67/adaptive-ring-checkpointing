# Codebase Structure

**Analysis Date:** 2026-04-18

## Directory Layout

```
adaptive-ring-checkpointing/
├── .planning/                # GSD planning documents
├── .venv/                    # Python virtual environment
├── checkpoints/              # Saved model checkpoints (*.pt files)
├── configs/                  # YAML configuration files
├── data/                     # Dataset storage (CIFAR-100, FaceForensics)
├── scripts/                  # Training entry points
├── src/                      # Core library code
│   └── coci/                 # Main package (Checkpointing, Orchestration, CI)
│       ├── checkpointing/    # Checkpoint strategies and management
│       ├── data_ingestor/    # Dataset loading (CIFAR, FaceForensics)
│       ├── fault/            # Fault injection utilities
│       ├── hashing/          # Hash ring and fault detection
│       └── models/           # Neural network architectures
└── pyproject.toml            # Project configuration
```

## Directory Purposes

**src/coci (Core Package):**
- Purpose: Main library containing all training and fault-tolerance components
- Contains: Python modules for distributed training, checkpointing, hashing
- Key files: `config.py`, `distributed.py`

**src/coci/checkpointing:**
- Purpose: Model state persistence and adaptive checkpoint strategies
- Contains: `checkpoint_manager.py`, `strategy.py`
- Key files: `src/coci/checkpointing/checkpoint_manager.py`, `src/coci/checkpointing/strategy.py`

**src/coci/data_ingestor:**
- Purpose: Dataset loading for CIFAR-100 and FaceForensics++
- Contains: `dataset.py`, `cifar.py`, `faceforensics.py`
- Key files: `src/coci/data_ingestor/dataset.py`, `src/coci/data_ingestor/cifar.py`

**src/coci/fault:**
- Purpose: Fault injection for fault tolerance testing
- Contains: `fault_injector.py`
- Key files: `src/coci/fault/fault_injector.py`

**src/coci/hashing:**
- Purpose: Distributed cache sharding with consistent hashing
- Contains: `hash_ring.py`, `fault_detector.py`, `recovery_scheduler.py`
- Key files: `src/coci/hashing/hash_ring.py`, `src/coci/hashing/fault_detector.py`

**src/coci/models:**
- Purpose: Neural network model architectures
- Contains: `model.py` (EfficientNet, ResNet, MobileNet, MultiFrameModel)
- Key files: `src/coci/models/model.py`

**scripts:**
- Purpose: Training entry points (single-GPU and distributed)
- Contains: `train.py`, `train_distributed.py`, `train_faceforensics.py`
- Key files: `scripts/train.py`, `scripts/train_distributed.py`

**configs:**
- Purpose: YAML configuration files for different modes
- Contains: `dev.yaml`, `server.yaml`
- Key files: `configs/dev.yaml`, `configs/server.yaml`

## Key File Locations

**Entry Points:**
- `scripts/train.py`: Single-GPU training with fault injection
- `scripts/train_distributed.py`: Multi-GPU DDP training via torchrun

**Configuration:**
- `src/coci/config.py`: Config dataclass and YAML loader
- `configs/dev.yaml`: Development configuration
- `configs/server.yaml`: Production/server configuration

**Core Logic:**
- `src/coci/distributed.py`: Distributed utilities (rank, barrier, reduce_metrics)
- `src/coci/checkpointing/checkpoint_manager.py`: Checkpoint save/load with DDP support
- `src/coci/checkpointing/strategy.py`: CheckpointStrategy abstract class and factory
- `src/coci/models/model.py`: Model factory and MultiFrameModel for video classification

**Testing/Verification:**
- `VERIFICATION_E2E.md`: End-to-end verification plan
- `VERIFICATION_RESULTS.md`: Verification test results
- `HASH_RING.md`: Hash ring implementation documentation

## Naming Conventions

**Files:**
- snake_case: `checkpoint_manager.py`, `fault_injector.py`, `hash_ring.py`
- Module prefix indicates domain: `*_distributed.py`, `*_strategy.py`

**Directories:**
- snake_case: `data_ingestor/`, `checkpointing/`, `fault/`
- Underscore separates words: `src/coci/`

**Classes:**
- PascalCase: `CheckpointManager`, `HashRing`, `FaultDetector`, `MultiFrameModel`
- Enum values: PascalCase (e.g., `NodeStatus.ALIVE`)

**Functions:**
- snake_case: `get_model()`, `setup_distributed()`, `should_checkpoint()`
- Prefix indicates purpose: `get_*` (factory), `is_*` (predicate), `load_*` / `save_*` (persistence)

**Variables:**
- snake_case: `checkpoint_dir`, `failure_rate_per_second`, `virtual_nodes_per_physical`
- Prefix indicates scope: `_private` (module-private), `is_*` (boolean)

## Where to Add New Code

**New Checkpoint Strategy:**
- Implementation: `src/coci/checkpointing/strategy.py`
- Extend `CheckpointStrategy` abstract class
- Register in `CheckpointStrategyFactory.create()`

**New Model Architecture:**
- Implementation: `src/coci/models/model.py`
- Add factory case in `get_model()` function

**New Dataset:**
- Implementation: `src/coci/data_ingestor/`
- Create new module (e.g., `imagenet.py`)
- Implement `torch.utils.data.Dataset` interface

**New Fault Type:**
- Implementation: `src/coci/fault/`
- Extend or create new injector class

**New Hash Ring Feature:**
- Implementation: `src/coci/hashing/`
- Extend `HashRing`, `ShardManager`, or `FaultDetector` classes

**New Training Script:**
- Location: `scripts/`
- Follow pattern of `train.py` or `train_distributed.py`

**New Configuration:**
- Location: `configs/`
- Add new YAML file following schema in `src/coci/config.py`

## Special Directories

**checkpoints/:**
- Purpose: Model checkpoint storage during training
- Generated: Yes (at runtime)
- Committed: No (in .gitignore)

**data/:**
- Purpose: Dataset storage
- Generated: Partially (CIFAR-100 downloads, FaceForensics++ if used)
- Committed: No (in .gitignore)

**.venv/:**
- Purpose: Python virtual environment
- Generated: Yes (via uv pip or venv)
- Committed: No (in .gitignore)

**src/coci/preprocessing/:**
- Purpose: Preprocessing scripts (currently empty, tracked in .gitinclude)
- Generated: No
- Committed: Skeleton only

---

*Structure analysis: 2026-04-18*