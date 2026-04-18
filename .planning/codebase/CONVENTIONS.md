# Coding Conventions

**Analysis Date:** 2026-04-18

## Naming Patterns

**Files:**
- snake_case: `hash_ring.py`, `fault_detector.py`, `checkpoint_manager.py`
- No specific prefix/suffix pattern observed

**Classes:**
- PascalCase: `HashRing`, `CheckpointManager`, `FaultDetector`, `ElasticRecaching`
- Factory suffix: `CheckpointStrategyFactory`

**Functions:**
- snake_case: `get_rank()`, `setup_distributed()`, `load_config()`, `get_efficientnet_binary()`
- Verb-noun pattern: `save()`, `load_latest()`, `maybe_fail()`

**Variables:**
- snake_case: `virtual_nodes_per_physical`, `checkpoint_dir`, `last_checkpoint_time`
- Underscores for readability in compound names

**Types:**
- PascalCase for type hints: `Dict[str, List[Optional[str]]`
- Dataclasses for config: `@dataclass`

## Code Style

**Formatting:**
- No automated formatter configured (no black, ruff, yapf)
- Manual 4-space indentation
- Line length: not enforced

**Linting:**
- No linting tool configured (no flake8, pylint, ruff)
- No .cfg or .toml lint config found

**Type Hints:**
- Partial usage - present in some files but not all
- Examples: `src/coci/config.py` uses complete type hints
- `src/coci/hash_ring.py` uses basic type hints: `Dict`, `List`, `Optional`, `Tuple`

## Import Organization

**Order:**
1. Standard library: `import os`, `import time`, `import hashlib`
2. Third-party: `import torch`, `import torch.nn as nn`
3. Local imports: `from src.coci.models.model import get_model`

**No explicit grouping:**
- No separator comments used
- No import sorting enforced

**Path Aliases:**
- None defined - absolute imports from package root

```python
# Standard pattern observed
import os
import torch
import torch.distributed as dist

from src.coci.models.model import get_model
from src.coci.config import load_config
```

## Error Handling

**Patterns:**
- Return early patterns: `if not condition: return`
- Explicit RuntimeError for failures:
```python
raise RuntimeError(f"Injected Failure on rank {self.rank}")
```
- Error messages include context: `f"Injected Failure on rank {self.rank}"`

**No try-except blocks observed in core logic:**
- Errors propagate to caller
- Training script catches `RuntimeError` at top level

## Logging

**Framework:** `print()` statements (not logging module)

**Patterns:**
- Prefix-based logging:
```python
print(f"[Checkpoint] Saved epoch {epoch} | Time: {duration:.4f}s")
print(f"[YoungDaly] Optimal interval: {self.optimal_interval:.2f} sec")
print("💥 Poisson failure triggered on rank {self.rank}!")
```

**Rank-aware logging:**
```python
def log_on_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)
```

**No structured logging:**
- No use of `logging` module
- No log levels
- No output formatting standardization

## Comments

**When to Comment:**
- Class docstrings: `"""Description.""""`
- Public method docstrings with Args/Returns:
```python
def get_rank():
    """
    Get the rank of the current process in the distributed group.

    Returns:
        int: Rank of the current process (0 if not distributed).
    """
```

**In-line comments:**
- Section dividers:
```python
# -------------------------------------------------
# Training Loop
# -------------------------------------------------
```
- Explanatory comments:
```python
# Use model.module.state_dict() if DDP wrapped, else model.state_dict()
# This handles the extra wrapper layer that DDP adds
```

**No TODO/FIXME comments found:**
- No inline technical debt markers

## Function Design

**Size:** No enforced limit, but functions average 20-50 lines

**Parameters:**
- Type hints optional: `def __init__(self, checkpoint_dir="checkpoints", is_ddp_wrapped=False):`
- Default values for optional params

**Return Values:**
-Tuple returns: `return rank, world_size, local_rank`
- Explicit type hints when complex: `-> Tuple[int, int, ResumeMode]`

## Module Design

**Single class per file:**
- Files named after class: `hash_ring.py` → `HashRing`
- One public class per module

**Barrel files:**
- Not used (no `__init__.py` re-exports observed)

**Exports:**
- All definitions at module level
- No `__all__` defined

## Class Structure

**Initialization:**
- Docstring in `__init__` if complex logic
- Instance attributes defined inline

**Inheritance:**
```python
class CheckpointStrategy(ABC):
    @abstractmethod
    def should_checkpoint(self, **kwargs) -> bool:
        pass
```
- ABC for abstract base classes

**Factory Pattern:**
```python
class CheckpointStrategyFactory:
    @staticmethod
    def create(cfg, checkpoint_cost=None, mtbf=None):
        if cfg.strategy == "fixed":
            return FixedIntervalStrategy(cfg.fixed_interval)
        # ...
```

## Data Structure Patterns

**Enums:**
```python
class NodeStatus(Enum):
    ALIVE = "ALIVE"
    SUSPECTED = "SUSPECTED"
    DEAD = "DEAD"
```

**Dicts for complex state:**
```python
self.shard_table: Dict[str, Dict] = {}
self.node_status: Dict[str, NodeStatus] = {}
```

**Thread safety:**
```python
self._lock = threading.Lock()

def mark_alive(self, node_id: str) -> None:
    with self._lock:
        # thread-safe operations
```

---

*Convention analysis: 2026-04-18*