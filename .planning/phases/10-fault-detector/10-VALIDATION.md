---
phase: 10
slug: fault-detector
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-18
---

# Phase 10 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | existing scripts/ |
| **Quick run command** | `python -m pytest scripts/ -v -k fault or python -c "[inline tests]"` |
| **Full suite command** | E2E verification scripts |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run quick inline tests
- **After every plan wave:** Run full fault detection integration
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 10-01-01 | 01 | 1 | FLTD-01 (TTL timeout) | unit | inline python | ⚠️ | ⬜ pending |
| 10-01-02 | 01 | 1 | FLTD-02 (node status) | unit | inline python | ⚠️ | ⬜ pending |
| 10-01-03 | 01 | 1 | FLTD-03 (gossip broadcast) | unit | inline python | ⚠️ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] Inline tests cover FaultDetector class (can use inline python -c for quick verification)
- [ ] Existing test infrastructure covers basic functionality
- [ ] No new framework dependencies

*Note: Non-distributed mode tests use inline Python to verify basic functionality without requiring torchrun.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Distributed gossip broadcast | FLTD-03 | Requires torchrun with multiple ranks | Run with `torchrun` or E2E test |

*All non-distributed behaviors have automated inline verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or inline tests
- [x] Sampling continuity: inline tests provide quick feedback
- [x] No new framework dependencies required
- [x] No watch-mode flags
- [x] Feedback latency < 30s

**Approval:** pending