# Context: Phase 13 - Experiment Results Organization

## Problem Statement

The `experiment_results/` directory has become disorganized with 20+ experiment directories using timestamp-based naming:
- `faceforensics_convergence_20260419_104954`
- `faceforensics_epoch_20260419_101309`
- `faceforensics_deepspeed_20260430_175303`
- etc.

This makes it difficult to:
- Find experiments by training mode (epoch vs convergence vs hash-ring)
- Compare results across different frameworks (vanilla PyTorch, DeepSpeed, FSDP, Lightning, HF)
- Locate experiments by date
- Understand what each experiment was testing

## Current Structure Analysis

**Experiment Types Found:**
1. **By Training Mode**: epoch, convergence, hash-ring-epoch, convergence-hash-ring
2. **By Framework**: vanilla PyTorch, DeepSpeed, FSDP, PyTorch Lightning, HuggingFace Trainer
3. **By Feature**: wandb-artifacts, hash-ring variants

**Files per Experiment:**
- `epoch_*.html` - Training reports
- `epoch_*.json` - Metrics data
- `epoch_*.md` - Markdown summaries
- `epoch_*.csv` - CSV exports

**Available Metadata:**
- `experiment_summaries.jsonl` - 20 entries with full metrics
- Each entry has: timestamp, experiment_name, training_mode, dataset, model, epochs, batch_size, performance metrics, world_size, cache metrics, fault injection results

## Requirements

### User Requirements (from conversation)
- Organize experiment results efficiently
- Make results easy to find and compare
- Maintain all historical data (no deletion)

### Implicit Requirements
- New experiments should follow the organization scheme
- Should be easy to compare similar experiments
- Directory structure should be intuitive
- Metadata/index should be maintainable

## Proposed Organization Strategy

### New Directory Structure (Hierarchical)
```
experiment_results/
├── by_mode/                    # Primary organization by training mode
│   ├── epoch/
│   │   ├── vanilla/           # vanilla PyTorch
│   │   ├── deepspeed/
│   │   ├── fsdp/
│   │   ├── lightning/
│   │   └── hf_trainer/
│   ├── convergence/
│   │   ├── vanilla/
│   │   └── hash_ring/         # convergence + hash ring
│   └── hash_ring_epoch/       # hash ring + epoch based
│       └── vanilla/
├── by_date/                   # Secondary organization (symlinks or copies)
│   ├── 2026-04-19/
│   └── 2026-04-30/
└── index/                     # Metadata and indexes
    ├── experiment_summaries.jsonl    # Existing (will be updated)
    ├── README.md                    # Documentation of structure
    └── lookup.json                  # Quick lookup by various criteria
```

### Alternative: Flat with Better Naming
```
experiment_results/
├── epoch_vanilla_20260419_101309/
├── convergence_vanilla_20260419_104954/
├── convergence_hashring_20260419_113143/
├── epoch_hashring_20260419_112016/
├── deepspeed_20260430_175303/
└── ...
```

## Decision Needed

**User Preference:** [To be confirmed]
- Option A: Hierarchical structure (by_mode/by_framework)
- Option B: Flat structure with improved naming convention
- Option C: Hybrid (hierarchical + symlinks for date-based access)

Recommendation: **Option A (Hierarchical)** as it provides best organization for comparison and analysis.

## Success Criteria

1. All 20 existing experiments are moved to new structure
2. No data is lost during migration
3. New structure is documented (README.md)
4. experiment_summaries.jsonl is updated with new paths
5. Helper script created for future experiments to use correct naming
6. Easy to find experiments by: mode, framework, date

## Constraints

- Must work on Windows (the development environment)
- Preserve all existing data
- Keep experiment_summaries.jsonl updated
- Minimize manual work for future experiments
