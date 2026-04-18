# Phase 10: Fault Detector - Research

**Researched:** 2026-04-18
**Domain:** Distributed systems failure detection, gossip protocols, PyTorch distributed
**Confidence:** HIGH

## Summary

Phase 10 implements timeout-based failure detection with gossip quorum for the adaptive ring checkpointing system. A `FaultDetector` class already exists in `src/coci/hashing/fault_detector.py` with basic functionality. This research identifies enhancements needed for distributed gossip-based failure broadcast and quorum confirmation.

**Primary recommendation:** Extend existing `FaultDetector` class with PyTorch distributed primitives for gossip communication, integrate with `HashRing` for node management, and add async heartbeat tracking for production use.

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FLTD-01 | Timeout-based failure detection with configurable TTL and timeout_count | ✓ Existing class has ttl_seconds, timeout_limit, heartbeat_interval |
| FLTD-02 | Node status tracking (ALIVE \| SUSPECTED \| DEAD) | ✓ Existing NodeStatus enum with all three states |
| FLTD-03 | Gossip-based failure broadcast to peers | ⚠ Needs enhancement with torch.distributed |
| FLTD-04 | Quorum-based failure confirmation (51% required) | ✓ Existing suspicion_quorum_pct = 51 |

## Standard Stack

### Core Dependencies
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | ≥2.0 | Distributed communication | Already project dependency |
| torch.distributed | (via torch) | Gossip primitives | For peer broadcast |

### Existing Implementation
| Component | Location | Status |
|-----------|----------|--------|
| FaultDetector | src/coci/hashing/fault_detector.py | ✓ Basic implementation exists |
| HashRing | src/coci/hashing/hash_ring.py | ✓ Phase 8 complete |
| ShardManager | src/coci/hashing/hash_ring.py | ✓ Phase 9 complete |

**Installation:**
No additional packages required - uses existing torch installation.

## Architecture Patterns

### Current Implementation (Existing)
The existing `FaultDetector` class provides:

```python
# Source: src/coci/hashing/fault_detector.py
class FaultDetector:
    def __init__(
        self,
        ttl_seconds: float = 5.0,
        timeout_limit: int = 3,
        heartbeat_interval: float = 1.0,
        suspicion_quorum_pct: int = 51,
    ):
        self.node_status: Dict[str, NodeStatus] = {}
        self.timeout_count: Dict[str, int] = {}
        self.last_heartbeat: Dict[str, float] = {}
        self.suspicion_votes: Dict[str, List[str]] = {}
```

### Recommended Pattern: Gossip Integration
For distributed gossip-based failure detection, extend with torch.distributed:

```python
# Pseudocode based on distributed.py patterns
import torch.distributed as dist

class DistributedFaultDetector(FaultDetector):
    def __init__(self, ..., rank: int, world_size: int):
        super().__init__(...)
        self.rank = rank
        self.world_size = world_size

    def broadcast_status(self):
        """Broadcast local node status to all peers via all_reduce."""
        status_tensor = torch.tensor([self._get_status_code()])
        dist.all_reduce(status_tensor, op=dist.ReduceOp.SUM)

    def gather_suspicions(self):
        """Gather suspicion votes from all peers for quorum calculation."""
        # Use all_gather to collect votes from all ranks
```

### Project Structure
```
src/coci/hashing/
├── hash_ring.py        # Phase 8 - HashRing class
├── fault_detector.py   # Phase 10 - FaultDetector (existing, needs enhancement)
└── __init__.py
```

### Anti-Patterns to Avoid
- **Centralized detection:** Don't use rank 0 as single point of failure detection - use decentralized gossip
- **Blocking collectives:** Use non-blocking async operations for heartbeats to avoid training stalls
- **Fixed timeouts without tuning:** TTL must be configurable per deployment (LAN vs WAN)

## Don't Build

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|------------|-----|
| Failure detection | Custom UDP server | torch.distributed collectives | Already initialized, works with NCCL |
| Node health API | HTTP health checks | Gossip status exchange | Works across all training steps |
| Leader election | Paxos/Raft | Quorum-based voting | Simpler, sufficient for our use case |
| Failure confirmation | SWIM indirect probing | Simple quorum (51%) | Matches FLTD-04 requirement |

**Key insight:** PyTorch distributed provides reliable message passing; leverage existing process group rather than building custom networking layer.

## Common Pitfalls

### Pitfall 1: Gossip Overhead
**What goes wrong:** Sending full membership list every heartbeat causes O(n²) bandwidth
**Why it happens:** Naive gossip sends entire node table each round
**How to avoid:** Send only delta updates (status changes), use lazy propagation (every N steps)
**Warning signs:** Training slows noticeably when adding more GPUs

### Pitfall 2: Blocking Detection
**What goes wrong:** Failure detection blocks training loop
**Why it happens:** Using synchronous dist.barrier() for status sync
**How to avoid:** Use async/timer-based detection separate from training loop
**Warning signs:** Training hangs during epochs

### Pitfall 3: Split Brain
**What goes wrong:** Different nodes have different views of dead nodes
**Why it happens:** Network partition causes some nodes to miss gossip
**How to avoid:** Requires quorum (51%) for DEAD status - matches FLTD-04
**Warning signs:** Inconsistent shard ownership after recovery

### Pitfall 4: Stale TTL
**What goes wrong:** Fixed TTL fails in variable network conditions
**Why it happens:** Without adaptive timeout, GC pauses or network jitter triggers false failures
**How to avoid:** Make TTL configurable, default conservative (5s for LAN), adjust per deployment

## Code Examples

### Example 1: Existing Basic Usage
```python
# Source: src/coci/hashing/fault_detector.py
from coci.hashing.fault_detector import FaultDetector, NodeStatus

detector = FaultDetector(
    ttl_seconds=5.0,
    timeout_limit=3,
    heartbeat_interval=1.0,
    suspicion_quorum_pct=51,
)

# Register node
detector.register_node("rank:0")
detector.register_node("rank:1")

# Mark alive
detector.mark_alive("rank:0")

# Check for timeouts
detector.check_timeout("rank:1")

# Get status
status = detector.get_status("rank:1")  # NodeStatus.ALIVE | SUSPECTED | DEAD
dead_nodes = detector.get_dead_nodes()
```

### Example 2: Integration with HashRing
```python
# Suggested integration pattern
hash_ring = HashRing(virtual_nodes_per_physical=100)
shard_manager = ShardManager(hash_ring=hash_ring)
fault_detector = FaultDetector()

# Register all nodes from hash ring
for node_id in hash_ring.get_nodes():
    fault_detector.register_node(node_id)

# On failure detection
def on_node_dead(node_id: str):
    # Update hash ring
    hash_ring.remove_node(node_id)
    # Orphan shards
    orphaned = shard_manager.get_orphaned_shards(node_id)
    for shard_id in orphaned:
        shard_manager.update_status(shard_id, "ORPHANED")
    # Trigger recaching (Phase 11)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Centralized monitor | Decentralized gossip | Traditional | Reduces single point of failure |
| Binary alive/dead | 3-state (ALIVE/SUSPECTED/DEAD) | SWIM 2002 | Reduces false positives |
| Fixed timeout | Configurable TTL + quorum | Modern | Balances detection speed vs accuracy |
| Blocking sync | Async heartbeat | Modern | Doesn't block training |

**Deprecated/outdated:**
- SWIM indirect probing: More complex, overkill for small GPU clusters
- Phi accrual failure detector: Statistical approach, requires tuning per environment

## Open Questions

1. **Gossip frequency vs training overhead:**
   - What we know: Full gossip each step is expensive
   - What's unclear: Optimal heartbeat_interval for 4-8 GPU training
   - Recommendation: Start with 1.0s (existing default), tune based on testing

2. **Integration with training loop:**
   - What we know: Need async heartbeat to avoid blocking
   - What's unclear: Best pattern (threading vs separate process group)
   - Recommendation: Use threading with timer, integrate at epoch boundary

3. **Recovery action on death:**
   - What we know: Need to trigger shard recaching (Phase 11)
   - What's unclear: Whether FaultDetector should notify or simply expose state
   - Recommendation: Expose state via get_dead_nodes(), let ElasticRecaching poll

## Sources

### Primary (HIGH confidence)
- Existing fault_detector.py - Basic implementation verified
- Existing distributed.py - Patterns for torch.distributed integration
- PyTorch DDP documentation - For collective primitives

### Secondary (MEDIUM confidence)
- SWIM Protocol (Cornell 2002) - Academic foundation for gossip protocols
- Gossip-style membership systems (Python implementations) - Practical patterns
- PyTorch HealthcheckNCCL PR - Similar failure detection in PyTorch core

### Tertiary (LOW confidence)
- Web search results on Phi Accrual detectors - Marked for validation in real testing

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (via project test scripts) |
| Config file | N/A - use existing scripts/ |
| Quick run command | `python -m pytest scripts/*fault* -v` or E2E test |
| Full suite command | Full verification after all v1.2 phases |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FLTD-01 | TTL-based timeout detection | unit | Import and test timeout logic | ❌ Create test |
| FLTD-02 | Node status state machine | unit | Test status transitions | ❌ Create test |
| FLTD-03 | Gossip broadcast | integration | Test distributed gather | ❌ Create test |
| FLTD-04 | Quorum confirmation | unit | Test 51% threshold | ❌ Create test |

### Sampling Rate
- **Per task commit:** Quick unit test for modified behavior
- **Per wave merge:** Integration test with multiple ranks
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `tests/test_fault_detector.py` — unit tests for FaultDetector class
- [ ] `tests/test_fault_detector_distributed.py` — integration with torch.distributed
- [ ] E2E test updates for fault detection flow

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Uses existing torch and PyTorch distributed
- Architecture: HIGH - Basic implementation exists, enhancement minimal
- Pitfalls: MEDIUM - Requires distributed testing for validation

**Research date:** 2026-04-18
**Valid until:** 2026-05-18 (30 days - stable domain)