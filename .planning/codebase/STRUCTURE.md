# Codebase Structure

**Analysis Date:** 2026-03-16

## Directory Layout

```
adaptive-ring-checkpointing/
├── .git/                     # Git repository
├── .venv/                    # Python virtual environment
├── .gitignore
├── .python-version           # Python version specification
├── pyproject.toml            # Project metadata and dependencies
├── requirements.txt          # Locked dependencies
├── README.md
├── experiment_log*.jsonl     # Experiment result logs
├── configs/
│   ├── dev.yaml              # Development configuration
│   └── server.yaml           # Server/production configuration
├── data/                     # Dataset storage
│   └── cifar-100-python/     # CIFAR-100 dataset
├── scripts/
│   └── train.py              # Main training entry point
└── src/
    ├── main.py               # Placeholder entry point
    └── coci/                 # Core package
        ├── __pycache__/      # Compiled Python
        ├── config.py         # Configuration loader
        ├── models/           # Model definitions
        │   └── model.py
        ├── data_ingestor/    # Dataset loading
        │   ├── cifar.py
        │   ├── dataset.py
        │   └── preprocessing/
        ├── checkpointing/    # Checkpoint management
        │   ├── checkpoint_manager.py
        │   └── strategy.py
        └── fault/            # Fault injection
            └── fault_injector.py
```

## Directory Purposes

**Root Directory:**
- Purpose: Project root containing all files
- Contains: Configuration files, dependencies, experiment logs

**src/:**
- Purpose: Source code directory
- Contains: Python modules
- Key files: `main.py`

**src/coci/:**
- Purpose: Core package containing all ML components
- Contains: config, models, data_ingestor, checkpointing, fault modules

**scripts/:**
- Purpose: Executable entry points
- Contains: `train.py`
- Key files: `train.py` - main training script

**configs/:**
- Purpose: Environment-specific configurations
- Contains: YAML config files
- Key files: `dev.yaml`, `server.yaml`

**data/:**
- Purpose: Dataset storage (not committed to git)
- Contains: CIFAR-100 dataset files

## Key File Locations

**Entry Points:**
- `scripts/train.py`: Main training script (primary entry)
- `src/main.py`: Placeholder/development entry

**Configuration:**
- `src/coci/config.py`: Config dataclass and loader
- `configs/dev.yaml`: Development settings
- `configs/server.yaml`: Production settings

**Core Logic:**
- `src/coci/models/model.py`: Model factory (ResNet18, ResNet50, MobileNet)
- `src/coci/checkpointing/checkpoint_manager.py`: Checkpoint save/load
- `src/coci/checkpointing/strategy.py`: Checkpoint timing strategies
- `src/coci/data_ingestor/cifar.py`: CIFAR-100 dataset loader
- `src/coci/fault/fault_injector.py`: Fault injection for testing

**Testing:**
- Not detected - no test directory or test files found

## Naming Conventions

**Files:**
- Python modules: `snake_case.py`
- Config files: `snake_case.yaml`

**Directories:**
- Package directories: `snake_case/`
- Sub-packages under coci: `models/`, `data_ingestor/`, `checkpointing/`, `fault/`

**Functions:**
- snake_case: `get_model()`, `load_config()`, `should_checkpoint()`, `maybe_fail()`

**Classes:**
- PascalCase: `CheckpointManager`, `CheckpointStrategy`, `FaultInjector`, `ImageDataset`

**Variables:**
- snake_case: `checkpoint_dir`, `num_checkpoints`, `failure_rate_per_second`

## Where to Add New Code

**New Checkpoint Strategy:**
- Implementation: `src/coci/checkpointing/strategy.py`
- Add new class extending `CheckpointStrategy`
- Register in `CheckpointStrategyFactory.create()`

**New Model:**
- Implementation: `src/coci/models/model.py`
- Add new model in `get_model()` function

**New Data Loader:**
- Implementation: `src/coci/data_ingestor/`
- Add new module (e.g., `imagenet.py`)
- Import in `scripts/train.py`

**New Fault Type:**
- Implementation: `src/coci/fault/`
- Add new class extending fault behavior
- Pass to `train()` function

**Configuration:**
- Add to: `configs/dev.yaml` or `configs/server.yaml`
- Update: `src/coci/config.py` dataclass

**New Experiment:**
- Run: `python scripts/train.py --mode dev --inject_fault`
- Results: Written to `experiment_log*.jsonl`

## Special Directories

**src/coci/preprocessing/:**
- Purpose: Data preprocessing utilities
- Contains: `.gitinclude` marker (empty/incomplete)
- Generated: No
- Committed: Partial (gitinclude only)

**data/:**
- Purpose: Dataset storage
- Contains: CIFAR-100 dataset
- Generated: Yes (downloaded on first run)
- Committed: Partial (dataset files in .gitignore)

**.venv/:**
- Purpose: Python virtual environment
- Contains: Python packages
- Generated: Yes
- Committed: No (.gitignored)

---

*Structure analysis: 2026-03-16*
