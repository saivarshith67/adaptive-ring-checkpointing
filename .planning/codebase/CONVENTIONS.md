# Coding Conventions

**Analysis Date:** 2026-03-16

## Language

**Primary:** Python 3.10.2+

**Runtime:** CPython (using `.venv` virtual environment)

**Package Manager:** uv (lockfile: `uv.lock`)

## Naming Patterns

**Files:**
- snake_case: `checkpoint_manager.py`, `fault_injector.py`, `cifar.py`
- Single-word modules: `config.py`, `model.py`, `dataset.py`

**Classes:**
- PascalCase: `CheckpointManager`, `FaultInjector`, `CheckpointStrategy`, `ImageDataset`
- Factory classes: `CheckpointStrategyFactory`

**Functions:**
- snake_case: `get_model()`, `save()`, `load_latest()`, `should_checkpoint()`
- Private functions: Not detected (no underscore prefix convention observed)

**Variables:**
- snake_case: `checkpoint_dir`, `last_checkpoint_time`, `failure_count`
- Local variables: `images`, `labels`, `model`, `optimizer`

**Constants:**
- Not explicitly detected (no ALL_CAPS convention)

## Code Style

**Formatting:**
- No explicit formatter detected (no `.prettierrc`, `black`, `ruff`)
- Manual indentation: 4 spaces
- Line length: Variable, not enforced

**Linting:**
- No linting configuration detected (no `.pylintrc`, `.flake8`, `ruff.toml`)
- No pre-commit hooks

**Type Hints:**
- Partial usage: `def load_config(path: str) -> Config:`
- Dataclass with type annotations: `@dataclass class Config:`
- Union types: `dataset_limit: int | None` (Python 3.10+ syntax)

## Import Organization

**Order:**
1. Standard library: `import os`, `import time`, `import math`, `import random`
2. Third-party: `import torch`, `import torchvision`, `import cv2`
3. Local project: `from src.coci.models.model import get_model`

**Path Aliases:**
- Not detected (no sys.path manipulation or path aliases)

## Error Handling

**Patterns:**
- Raise with message: `raise ValueError("Unknown checkpoint strategy")`
- Runtime errors: `raise RuntimeError("Injected Failure")`
- No try/catch blocks in main code (errors propagate)
- In `scripts/train.py`: try/catch around main training block to capture RuntimeError from fault injection

**Logging:**
- Print-based: `print(f"[Checkpoint] Saved epoch {epoch} | Time: {duration:.4f}s")`
- Bracketed prefixes: `[Checkpoint]`, `[YoungDaly]`
- Emoji usage: `print("💥 Poisson failure triggered!")`

## Comments

**When to Comment:**
- Section headers: `# -------------------------------------------------` with label
- Inline explanations: `# Time-based checkpointing`, `# Fault Injection`
- Metrics tracking: `# 🔥 Metrics`, `# 🔥 Reset timer after resume`

**JSDoc/TSDoc:**
- Not used

**Inline:**
- Minimal, mostly for section delineation

## Function Design

**Size:**
- Variable; `CheckpointManager` has small focused methods
- `scripts/train.py` has monolithic `main()` function (295 lines)

**Parameters:**
- Positional with keyword options: `train(..., checkpoint_manager, strategy, fault_injector=None, inject_fault=False, start_epoch=0)`

**Return Values:**
- Explicit returns: `return 0`, `return True`, `return model`
- Some functions don't return (procedural)

## Module Design

**Exports:**
- No explicit `__all__` detected

**Barrel Files:**
- Not detected (no `__init__.py` in package directories)

**Package Structure:**
```
src/coci/
├── __init__.py        # Not present
├── config.py
├── checkpointing/
│   ├── checkpoint_manager.py
│   └── strategy.py
├── data_ingestor/
│   ├── cifar.py
│   └── dataset.py
├── fault/
│   └── fault_injector.py
└── models/
    └── model.py
```

## Design Patterns

**Strategy Pattern:**
- Abstract base class: `CheckpointStrategy(ABC)`
- Concrete implementations: `FixedIntervalStrategy`, `YoungDalyStrategy`, `EpochStrategy`
- Factory: `CheckpointStrategyFactory.create()`

**Dataclass:**
- `@dataclass class Config:` in `src/coci/config.py`

**Singleton-ish (implicit):**
- `CheckpointManager` instantiated once in `scripts/train.py`

## Additional Observations

**Magic Numbers:**
- Direct values: `lr=1e-3`, `num_classes=100`, `weights=None`

**Emoji in Code:**
- Present: `💥`, `🔥` used in print statements and comments

**Testing:**
- No test files detected
- No testing framework in dependencies

---

*Convention analysis: 2026-03-16*
