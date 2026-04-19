# Phase 9: Shard Manager - Research

**Researched:** 2026-04-18
**Domain:** Checkpoint shard lifecycle and NVMe caching
**Confidence:** HIGH

## Summary

Phase 9 implements checkpoint shard lifecycle management with local NVMe caching. Building on the existing `ShardManager` class from Phase 8 which has metadata tracking but no actual file I/O, this phase adds real cache read/write operations to complete SHrd-04.

**Key findings:**
1. The existing ShardManager has state machine transitions but no file operations - need to add actual read/write methods
2. For NVMe caching, use Python's standard `shutil` for synchronous checkpoint I/O (simpler, sufficient for training workload)
3. Async I/O libraries (aiofiles, aiofile) available but unnecessary for checkpoint-style workloads - writes are periodic and blocking is acceptable
4. Atomic writes via temp file + rename is the standard pattern for crash-safe checkpoint updates
5. Cache directory management should include space monitoring and cleanup of orphaned files

**Primary recommendation:** Extend existing ShardManager with:
- `cache_shard()` method for write with atomic rename
- `load_cached_shard()` method for read with validation
- `verify_shard()` method for integrity checking
- Cache space management

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SHrd-01 | Shard lifecycle state machine (UNCACHED → CACHING → CACHED → ORPHANED → RE-CACHED) | Already implemented in Phase 8 ShardManager |
| SHrd-02 | Shard metadata tracking (owner, status, paths, last_verified) | Already implemented in Phase 8 ShardManager |
| SHrd-03 | Shard assignment to nodes via hash ring | Already implemented in Phase 8 ShardManager |
| SHrd-04 | Local NVMe cache management for shards | This research - file I/O operations needed |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python shutil | stdlib | Atomic file moves/copies | Standard for safe checkpoint writes |
| Python tempfile | stdlib | Temporary file handling | Built-in atomic write pattern |
| pathlib | stdlib | Path object operations | Modern, cross-platform path handling |

### Async I/O (Optional/Not Needed)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| aiofiles | v25.1.0 | Async file operations | If checkpointing during training iterations needed |
| aiofile | v3.9.0 | libaio-based async | Lowest latency NVMe writes (Linux only) |

**Note:** Async I/O is NOT required for Phase 9. Checkpoint writes are periodic (every N epochs) and blocking is acceptable during epoch boundaries. The SC'24 paper's fast path is about avoiding PFS access, not async writes.

**Installation:**
```bash
# No additional packages needed - all features use Python stdlib
```

## Architecture Patterns

### Recommended Project Structure
```
src/coci/
├── hashing/
│   ├── __init__.py
│   ├── hash_ring.py      # Existing - HashRing, ShardManager (partial)
│   └── shard_cache.py   # NEW - ShardManager extensions with file I/O
```

### Pattern 1: Atomic Checkpoint Write
**What:** Write to temp file then atomic rename for crash-safe updates
**When to use:** Writing any checkpoint shard to NVMe cache
**Example:**
```python
from pathlib import Path
import tempfile
import shutil

def cache_shard(self, shard_id: str, state_dict: dict) -> bool:
    """
    Write shard to local NVMe cache with atomic rename.
    
    Pattern: write to temp -> fsync -> rename to final
    This ensures no partial/corrupt files on crash.
    """
    shard = self.shard_table.get(shard_id)
    if not shard:
        return False
    
    cached_path = Path(shard["cached_path"])
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(
        mode='wb',
        dir=cached_path.parent,
        delete=False
    ) as f:
        temp_path = f.name
        torch.save(state_dict, f)
        f.flush()
        os.fsync(f.fileno())  # Ensure data on disk
    
    # Atomic rename
    shutil.move(temp_path, cached_path)
    
    # Update status
    self.update_status(shard_id, "CACHED")
    return True
```

### Pattern 2: Shard Loading with Verification
**What:** Load cached shard and verify integrity
**When to use:** Reading shard from local cache
**Example:**
```python
import torch
from pathlib import Path

def load_cached_shard(self, shard_id: str) -> Optional[dict]:
    """
    Load shard from local NVMe cache with optional verification.
    """
    shard = self.shard_table.get(shard_id)
    if not shard:
        return None
    
    cached_path = Path(shard["cached_path"])
    if not cached_path.exists():
        self.update_status(shard_id, "ORPHANED")
        return None
    
    try:
        state_dict = torch.load(cached_path, map_location='cpu')
        self.update_status(shard_id, "CACHED")
        return state_dict
    except Exception as e:
        # Log error, mark as invalid
        return None
```

### Pattern 3: Cache Space Management
**What:** Monitor cache usage and cleanup orphaned files
**When to use:** Periodic cache maintenance
**Example:**
```python
from pathlib import Path
import shutil

def get_cache_usage(self) -> dict:
    """Return cache usage statistics."""
    cache_dir = Path(self.cache_dir)
    if not cache_dir.exists():
        return {"total_bytes": 0, "file_count": 0}
    
    files = list(cache_dir.glob("*.pt"))
    total_bytes = sum(f.stat().st_size for f in files)
    
    return {
        "total_bytes": total_bytes,
        "file_count": len(files),
        "cache_dir": str(cache_dir)
    }

def cleanup_orphaned_files(self, dead_node: str) -> int:
    """Remove cached files for orphaned shards."""
    orphaned = self.get_orphaned_shards(dead_node)
    removed = 0
    
    for shard_id in orphaned:
        shard = self.shard_table.get(shard_id)
        if shard:
            cached_path = Path(shard["cached_path"])
            if cached_path.exists():
                cached_path.unlink()
                removed += 1
    
    return removed
```

### Pattern 4: Shard Verification
**What:** Periodic verification of cached shard integrity
**When to use:** Health checks, before using cached shard
**Example:**
```python
import torch
from datetime import datetime

def verify_shard(self, shard_id: str) -> bool:
    """Verify shard integrity by loading and checking."""
    shard = self.shard_table.get(shard_id)
    if not shard:
        return False
    
    cached_path = Path(shard["cached_path"])
    if not cached_path.exists():
        return False
    
    try:
        state_dict = torch.load(cached_path, map_location='cpu')
        # Update last_verified timestamp
        self.shard_table[shard_id]["last_verified"] = datetime.now().isoformat()
        return True
    except Exception:
        return False
```

### Anti-Patterns to Avoid
- **Direct write to final path:** Never write directly to `cached_path` - crashes can leave corrupt files. Use temp + rename.
- **No fsync before rename:** Always call `fsync()` before rename to ensure data persists to disk.
- **Using async for checkpoint:** Don't add async I/O libraries unless profiling shows blocking is a problem. Training is I/O-bound in compute, not checkpoint.
- **No space checking:** Check available space before write to avoid partial writes.

## Code Examples

### Existing ShardManager (from hash_ring.py)
```python
# Already implemented - partial ShardManager
class ShardManager:
    ShardStatus = [
        "UNCACHED",
        "CACHING",
        "CACHED",
        "ORPHANED",
        "RE-CACHED",
    ]

    # Methods needed:
    # - register_shard() - DONE
    # - get_shard() - DONE
    # - get_owner() - DONE
    # - update_status() - DONE
    # - reassign_shard() - DONE
    # - get_orphaned_shards() - DONE
    # - get_all_shards() - DONE
    
    # Methods TO IMPLEMENT (Phase 9):
    # - cache_shard(shard_id, state_dict) -> bool
    # - load_cached_shard(shard_id) -> Optional[dict]
    # - verify_shard(shard_id) -> bool
    # - get_cache_usage() -> dict
    # - cleanup_orphaned_files(dead_node) -> int
```

### Transition between States
```python
# State machine transitions
def cache_shard(self, shard_id: str, state_dict: dict) -> bool:
    """UNCACHED -> CACHING -> CACHED"""
    self.update_status(shard_id, "CACHING")
    success = self._write_to_cache(shard_id, state_dict)
    if success:
        self.update_status(shard_id, "CACHED")
    return success

def mark_orphaned(self, shard_id: str) -> None:
    """CACHED -> ORPHANED (when owner dies)"""
    if self.shard_table[shard_id]["status"] == "CACHED":
        self.update_status(shard_id, "ORPHANED")

def recache_shard(self, shard_id: str, state_dict: dict) -> bool:
    """ORPHANED -> RE-CACHED"""
    self.update_status(shard_id, "CACHING")
    success = self._write_to_cache(shard_id, state_dict)
    if success:
        self.update_status(shard_id, "RE-CACHED")
    return success
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Crash-safe checkpoint writes | Custom atomic write logic | Python shutil + tempfile | Well-tested, handles edge cases |
| Cross-platform paths | String concatenation | pathlib.Path | Handles Windows/Linux differences |
| Large file handling | Custom streaming | torch.load() with mmap | Built into PyTorch |

**Key insight:** The checkpoint workload is not I/O-bound during training - it's compute-bound. The optimization in SC'24 paper is about avoiding PFS access (network latency), not about async I/O. Standard blocking writes are fine.

## Common Pitfalls

### Pitfall 1: Missing fsync Before Rename
**What goes wrong:** Power failure before fsync leaves empty/corrupt file
**Why it happens:** OS write buffering delays actual disk write
**How to avoid:** Always call `os.fsync()` before `shutil.move()`
**Warning signs:** Checkpoint loads fail after system crash but file exists

### Pitfall 2: No Space Checking
**What goes wrong:** Write fails mid-transfer due to full disk
**Why it happens:** NVMe fills up without monitoring
**How to avoid:** Check `shutil.disk_usage()` before write, implement cleanup policy
**Warning signs:** WriteError during checkpoint save

### Pitfall 3: Cache Directory Not Created
**What goes wrong:** FileNotFoundError when writing to new cache dir
**Why it happens:** `mkdir(parents=True)` not called
**How to avoid:** Create parent directories before first write
**Warning signs:** First checkpoint write fails

### Pitfall 4: Wrong map_location on Load
**What goes wrong:** CUDA tensors fail to load when GPU unavailable
**Why it happens:** Checkpoint saved with GPU tensors, loaded on different machine
**How to avoid:** Always use `map_location='cpu'` for loading from NVMe cache
**Warning signs:** RuntimeError about CUDA device

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Direct write + rename | Temp file + fsync + atomic rename | ~2015 | Crash-safe checkpoints |
| Blocking torch.save | Still blocking (ok for periodic writes) | Current | No need for async in training |
| String paths | pathlib.Path | Python 3.4+ | Cross-platform compatibility |

**Deprecated/outdated:**
- No longer using `torch.save(state_dict, filepath)` directly - must use temp + rename pattern
- Async I/O frameworks (aiofiles) not needed for training checkpoint workloads

## Open Questions

1. **Cache size limit policy**
   - What we know: Should monitor disk usage
   - What's unclear: LRU vs LFU vs epoch-based eviction
   - Recommendation: Start with simple "warn if > 80% full", evolve as needed

2. **Verification frequency**
   - What we know: Verification on load is good practice
   - What's unclear: Full verification (load + validate) vs metadata only
   - Recommendation: Full verify on load, periodic full scan

3. **Compression**
   - What we know: Checkpoints can be large (GB)
   - What's unclear: lz4 vs zstd for performance
   - Recommendation: No compression in v1 - add only if profiling shows bottleneck

## Validation Architecture

> Validation enabled per workflow.nyquist_validation in .planning/config.json

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (project standard) |
| Config file | none — use inline test functions |
| Quick run command | `python -m pytest tests/ -v -k shard` |
| Full suite command | `python -m pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SHrd-01 | State machine transitions | unit | `pytest tests/test_shard_manager.py::test_state_machine -x` | ❌ Wave 0 |
| SHrd-02 | Metadata tracking | unit | `pytest tests/test_shard_manager.py::test_metadata -x` | ❌ Wave 0 |
| SHrd-03 | Hash ring assignment | unit | `pytest tests/test_shard_manager.py::test_assignment -x` | ❌ Wave 0 |
| SHrd-04 | Cache read/write/verify | unit | `pytest tests/test_shard_manager.py::test_cache_ops -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_shard_manager.py -x`
- **Per wave merge:** `python -m pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_shard_manager.py` — covers SHrd-01, SHrd-02, SHrd-03, SHrd-04
- [ ] Sample checkpoint files for testing (created in test fixtures)

*(If no gaps: "None — existing test infrastructure covers all phase requirements")*

## Sources

### Primary (HIGH confidence)
- Python shutil documentation - atomic file operations
- Python pathlib documentation - path handling
- PyTorch torch.save/load documentation - checkpoint serialization

### Secondary (MEDIUM confidence)
- WebSearch: "FastPersist NVMe checkpoint PyTorch" - NVMe optimization patterns
- WebSearch: "aiofiles vs synchronous checkpoint" - when async is needed

### Tertiary (LOW confidence)
- WebSearch only: Various blog posts on checkpoint best practices

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Using Python stdlib, well-documented
- Architecture: HIGH - Extending existing ShardManager with standard patterns
- Pitfalls: HIGH - Common checkpoint issues, well-documented in literature

**Research date:** 2026-04-18
**Valid until:** 2026-05-18 (30 days - stable domain)