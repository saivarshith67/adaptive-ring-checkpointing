# Testing Patterns

**Analysis Date:** 2026-04-18

## Test Framework

**Current State:** No tests present in codebase

**No test infrastructure configured:**
- No pytest, unittest, or nose configuration
- No `pytest.ini`, `setup.cfg`, or `pyproject.toml` test sections
- No `tests/` directory or test files found

**Dependencies available for testing:**
- `torch` - includes `torch.testing` module
- `scikit-learn` - includes metrics for validation

**Recommended framework:** `pytest`
- Compatible with Python 3.10+
- Standard for PyTorch projects

## Test File Organization

**Location:**
```
project-root/
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures
│   ├── unit/
│   │   ├── test_hash_ring.py
│   │   ├── test_checkpoint_manager.py
│   │   └── test_fault_detector.py
│   ├── integration/
│   │   └── test_distributed.py
│   └── e2e/
│       └── test_training.py
├── scripts/
│   └── train.py
└── src/coci/
```

**Naming:**
- `test_<module>.py` pattern
- `Test<ClassName>` class pattern

**Structure:**
- Fixtures in `conftest.py`
- Shared utilities in test helpers module

## Test Structure

**Unit test pattern:**
```python
import pytest
from src.coci.hashing.hash_ring import HashRing, ShardManager

class TestHashRing:
    def test_add_node(self):
        ring = HashRing()
        ring.add_node("node1")
        assert ring.get_nodes() == ["node1"]
    
    def test_get_node_returns_owner(self):
        ring = HashRing()
        ring.add_node("node1")
        result = ring.get_node("key1")
        assert result == "node1"

    def test_get_node_empty_ring_returns_none(self):
        ring = HashRing()
        result = ring.get_node("any_key")
        assert result is None

@pytest.fixture
def hash_ring():
    return HashRing(virtual_nodes_per_physical=10)

@pytest.fixture
def shard_manager(hash_ring):
    return ShardManager(hash_ring=hash_ring)
```

**Integration test pattern:**
```python
import pytest
import torch
import torch.distributed as dist
from src.coci.distributed import setup_distributed, cleanup_distributed

def test_distributed_setup():
    rank, world_size, local_rank = setup_distributed()
    assert rank >= 0
    assert world_size >= 1
    cleanup_distributed()
```

## Mocking

**Framework to use:** `unittest.mock` (standard library)

**Patterns:**
```python
from unittest.mock import Mock, patch, MagicMock

# Mocking torch.distributed
@patch('torch.distributed.is_initialized')
def test_checkpoint_rank_check(is_init_mock):
    is_init_mock.return_value = False
    # test logic
    
# Mocking file operations
@patch('torch.save')
def test_save_checkpoint(save_mock, tmp_path):
    manager = CheckpointManager(checkpoint_dir=str(tmp_path))
    model = Mock()
    optimizer = Mock()
    manager.save(model, optimizer, 0, 0.5)

# Mocking network calls
@patch('requests.get')
```

**What to Mock:**
- File I/O: `torch.save()`, `torch.load()`, file operations
- Network: distributed primitives
- Time-dependent logic: `time.time()`

**What NOT to Mock:**
- Model architecture logic
- Core algorithm implementations

## Fixtures and Factories

**Test data:**
```python
import pytest
from src.coci.hashing.hash_ring import HashRing

@pytest.fixture
def sample_hash_ring():
    ring = HashRing(virtual_nodes_per_physical=100)
    for i in range(4):
        ring.add_node(f"node_{i}")
    return ring

@pytest.fixture
def mock_model_state():
    return {
        'epoch': 5,
        'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
        'optimizer_state_dict': {},
        'loss': 0.5,
        'metric': 0.85
    }
```

**Location:**
- Local fixtures in test files
- Shared fixtures: `tests/conftest.py`

## Coverage

**Requirements:** No coverage enforced

**Recommended:**
```bash
pytest --cov=src/coci --cov-report=html
```

**Coverage target:** 80%+ for core modules

**View coverage:**
```bash
# Unit tests with coverage
pytest tests/unit/ --cov=src/coci --cov-report=term-missing

# Run and view HTML report
pytest tests/ --cov=src/coci --cov-report=html
open htmlcov/index.html
```

## Test Types

**Unit Tests:**
- HashRing: add_node, remove_node, get_node, get_virtual_node_count
- ShardManager: register_shard, get_shard, update_status, reassign_shard
- CheckpointManager: save, load_latest (mocked I/O)
- FaultDetector: register_node, mark_alive, get_status
- CheckpointStrategy: should_checkpoint intervals

**Integration Tests:**
- Distributed setup/teardown
- Checkpoint save/load with DDP
- Fault injection triggering

**E2E Tests:**
- Full training loop with mock data
- Multi-GPU fault recovery (if resources available)

## Common Patterns

**Async Testing (for async code):**
```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_recovery():
    result = await recovery_scheduler.async_recover()
    assert result is not None
```

**Parameterised tests:**
```python
@pytest.mark.parametrize("strategy", [
    "fixed",
    "young_daly",
    "epoch"
])
def test_strategy_factory(strategy):
    from src.coci.checkpointing.strategy import CheckpointStrategyFactory
    cfg = Mock(strategy=strategy)
    result = CheckpointStrategyFactory.create(cfg)
    assert result is not None
```

**Error testing:**
```python
def test_checkpoint_strategy_unknown():
    cfg = Mock(strategy="unknown")
    with pytest.raises(ValueError, match="Unknown checkpoint strategy"):
        CheckpointStrategyFactory.create(cfg)
```

## Running Tests

**Commands:**
```bash
# All tests
pytest

# Specific directory
pytest tests/unit/

# With verbose output
pytest -v

# Stop on first failure
pytest -x

# Watch mode (requires pytest-watch)
ptw

# Specific test file
pytest tests/unit/test_hash_ring.py

# Run tests matching pattern
pytest -k "test_add"

# Parallel execution (requires pytest-xdist)
pytest -n auto
```

---

*Testing analysis: 2026-04-18*