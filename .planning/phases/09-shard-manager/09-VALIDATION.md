---
phase: 09
slug: shard-manager
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-18
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (inline test functions) |
| **Config file** | none — project has no test framework yet |
| **Quick run command** | `python -c "from src.coci.hashing import ShardManager, HashRing; print('OK')"` |
| **Full suite command** | TBD - tests directory does not exist |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick import test
- **After every plan wave:** Run quick import test
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 09-01-01 | 01 | 1 | SHrd-04 (cache_shard, load_cached_shard) | unit | `python -c "..."` | ❌ W0 | ⬜ pending |
| 09-01-02 | 01 | 1 | SHrd-04 (verify_shard, get_cache_usage) | unit | `python -c "..."` | ❌ W0 | ⬜ pending |
| 09-01-03 | 01 | 1 | SHrd-04 (cleanup_orphaned) | unit | `python -c "..."` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

**Note:** SHrd-01, SHrd-02, SHrd-03 already implemented in Phase 8 — verification not needed in Phase 9.

---

## Wave 0 Requirements

- [ ] Test framework setup (pytest installation) — if needed
- [ ] Quick validation commands for each task

*Note:* Project has no tests/ directory. Phase 9 focuses on adding file I/O operations to existing ShardManager.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Cross-GPU checkpoint load | SHrd-04 | Requires multi-GPU env | Load checkpoint saved on GPU-0, load on CPU |

---

## Validation Sign-Off

- [x] Phase 9 adds to existing ShardManager (no new classes needed)
- [x] Quick validation via Python import works
- [x] Only new operations are cache I/O (SHrd-04)
- [ ] `nyquist_compliant: true` set in frontmatter (after tests exist)