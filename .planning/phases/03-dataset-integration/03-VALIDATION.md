---
phase: 03
slug: dataset-integration
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-16
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | none — Wave 0 creates test infrastructure |
| **Quick run command** | `pytest tests/ -v` |
| **Full suite command** | `pytest tests/ -v --tb=short` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -v`
- **After every plan wave:** Run `pytest tests/ -v --tb=short`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 03-01-01 | 01 | 1 | DATA-04 | manual | kagglehub download | ❌ W0 | ⬜ pending |
| 03-01-02 | 01 | 1 | DATA-05 | unit | Dataset batch shape | ❌ W0 | ⬜ pending |
| 03-01-03 | 01 | 1 | DATA-06 | unit | MTCNN detection | ❌ W0 | ⬜ pending |
| 03-01-04 | 01 | 1 | DATA-07 | unit | Transform output | ❌ W0 | ⬜ pending |
| 03-01-05 | 01 | 1 | CKPT-04 | manual | Checkpoint save/load | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_dfd_dataset.py` — covers DATA-05, DATA-07
- [ ] `tests/test_mtcnn_detection.py` — covers DATA-06  
- [ ] `tests/test_checkpoint_ddp.py` — covers CKPT-04
- [ ] Framework install: `pip install pytest` — if tests desired
- [ ] `tests/conftest.py` — shared fixtures for dataset paths

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Dataset download via kagglehub | DATA-04 | Requires Kaggle credentials | Run: `python -c "import kagglehub; print(kagglehub.dataset_download('sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset'))"` |
| Checkpoint save/load with DDP | CKPT-04 | Requires GPU + DDP | Run training script with checkpoint save, then load and verify model resumes |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** {pending / approved YYYY-MM-DD}
