# COCI: Convergence-Aware Optimal Checkpointing Interval
## Methodology Spec Sheet

**Paper:** *Convergence-aware optimal checkpointing for exploratory deep learning training jobs*
**Authors:** Hongliang Li, Zichen Wang, Hairui Zhao, Meng Zhang, Xiang Li, Haixiao Xu — Jilin University
**Published in:** Future Generation Computer Systems 164 (2025) 107597
**Code:** https://github.com/wangzc-HPC/COCI.git

---

## 1. Core Problem

Traditional HPC checkpointing (e.g., Young's formula, CCM) assumes job progress is **linear with execution time**, so fixed-interval checkpoints protect all stages equally. This assumption breaks down for DL training jobs for three reasons:

1. **Diminishing returns** — Loss reduction is largest in early iterations; later iterations yield smaller improvements.
2. **Exploratory nature** — Over 55% of DL jobs are terminated early (typically at 40–60% of total runtime) as practitioners tune hyperparameters. Early-stage progress is therefore disproportionately valuable.
3. **Quality-driven scheduling** — State-of-the-art schedulers (e.g., SLAQ) allocate more GPU workers to early training stages, making early failures proportionally more costly in parallel settings.

**Conclusion:** Early training stages are more valuable and deserve denser checkpoint protection than later stages.

---

## 2. Key Insight

Instead of partitioning training by **execution time** (fixed intervals), COCI partitions training by **convergence progress** (loss reduction). Each checkpoint segment covers an equal amount of loss reduction (`P_e`), producing shorter time intervals early (dense checkpoints) and longer intervals later (sparse checkpoints).

---

## 3. Mathematical Foundation

### 3.1 Loss Curve Model

The training loss is fit using an exponential decay function:

```
loss(t) = exp(θ₁·t + θ₂)
```

- `θ₁` < 0 — controls decay rate (shape of curve)
- `θ₂` — controls initial loss value (position of curve)
- Both coefficients are estimated online during training

### 3.2 Fault Tolerance Overhead (Objective Function)

Total fault tolerance overhead in terms of **progress loss**:

```
P_l = P_s + P_r
    = Σᵢ [ P_s[i] + P_r[i] ]
```

Where:
- `P_s[i]` = progress lost due to taking a checkpoint in interval `i`
- `P_r[i]` = progress lost due to recovery after a failure in interval `i`
- Failures follow a Poisson process with rate `λ = 1/MTBF`

### 3.3 Checkpoint Cost (in progress units)

Mapping checkpoint time cost `T_s` to progress space:

```
P_s[i] = (1 - exp(θ₁·Tₛ)) · (exp(θ₂) - i·P_e)
```

### 3.4 Full Overhead Expression

```
P_l = Σᵢ (1 - exp(θ₁·Tₛ))(exp(θ₂) - i·P_e)
    + Σᵢ ∫[i·Pe to (i+1)·Pe] (p - i·P_e)(λ·exp(-λ·loss⁻¹(p))) dp
```

### 3.5 Optimal Segment Size — Serial Case (Theorem 4.1)

Minimizing `P_l` with respect to `P_e` (by setting dP_l/dP_e = 0) yields:

```
P_e = (1 - exp(θ₁·Tₛ)) · (1/(2λ) - 1/(2θ₁))
```

This closed-form solution has **O(1) time complexity** and requires no user input — only `θ₁`, `λ`, and `Tₛ`.

### 3.6 Optimal Segment Size — Data Parallel Case (Theorem 4.2)

For each scheduling round `j` with `γⱼ` workers and variation coefficient `ξⱼ`:

```
Loss_j(t) = exp(θ₁,j · t + θ₂,j)   where θ₁,j = ξⱼ · θ₁

P_e,j = (1 - exp(ξⱼ·θ₁·γⱼ·Tₛ)) · (1/(2λ) - 1/(2ξⱼ·θ₁))
```

- `γⱼ` = number of workers in round `j`
- `ξⱼ` ≈ 1 — observed acceleration coefficient for round `j`
- Checkpoint cost scales to `γⱼ · Tₛ` in data parallel mode

---

## 4. Online Loss Curve Fitting: Piecewise Online Fitting (POF)

Because hyperparameter changes can cause abrupt shifts in the loss curve during exploratory training, COCI uses **Piecewise Online Fitting (POF)** rather than fitting all historical data at once.

**How POF works:**

1. Maintain two rolling arrays:
   - `iter_list[]` — iteration completion timestamps
   - `loss_list[]` — loss values at each iteration end
2. Periodically invoke `fit()` at intervals of `fit_interval` iterations
3. Fit `θ₁` and `θ₂` using only data from the **current fitting segment** (since the last coefficient change)
4. After each fit, check if coefficients have changed beyond thresholds:
   - `δθ₁ = 0.0001`
   - `δθ₂ = 0.1`
5. If thresholds exceeded, start a new fitting segment and recompute the checkpoint plan

**Two fitting modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| FPF (Fixed Parameter Fitting) | `θ₁`, `θ₂` fixed for entire run | Repeated training of same job with fixed hyperparameters |
| LOF (Loss-aware Online Fitting) | `θ₁`, `θ₂` updated dynamically via POF | Exploratory training with hyperparameter changes |

---

## 5. System Architecture: Three Modules

### 5.1 Fitter
- Runs FPF or LOF/POF to estimate `θ₁` and `θ₂`
- Invoked at configurable `fit_interval`
- Triggers checkpoint plan recomputation when coefficients drift past thresholds

### 5.2 Planner
- `general_profile()` — measures checkpoint file size and snapshot time cost (`Tₛ`) using a background thread during the first 50 iterations
- `parameter_profile()` — collects iteration timing and loss history; calls Fitter to get `θ₁`, `θ₂`
- Computes `P_e` using Theorem 4.1 / 4.2 (O(1) algorithm)
- Updates the checkpoint schedule for subsequent iterations
- Treats each checkpoint interval as **atomic** — plan changes only take effect at the next interval boundary

### 5.3 Manager
- `save()` — takes a two-phase asynchronous checkpoint (memory snapshot → persist), overlapped with training via a separate thread/process
- `recovery()` — maintains the last two checkpoint versions; restores model weights, optimizer states, and metadata via `_recovery()`

---

## 6. Implementation Steps

### Step 1: Profile Phase (first epoch)
- Profile for the first 50 iterations: measure `Tₛ` (checkpoint cost in seconds)
- Profile for the first `profile_threshold` iterations (default: 10,000): collect loss history, fit initial `θ₁` and `θ₂`
- No checkpoints are taken during this phase

### Step 2: Compute Initial `P_e`
Apply the closed-form formula:
```python
P_e = (1 - exp(θ₁ * T_s)) * (1 / (2 * λ) - 1 / (2 * θ₁))
```

### Step 3: Map `P_e` to Time Intervals
For each segment `i`, solve for the wall-clock duration `TC[i]` by inverting the loss function:
```
TC[i] = loss⁻¹(loss(t_i) - P_e) - t_i
```
where `t_i` is the start time of segment `i`. Segments are shorter early in training and grow longer as the curve flattens.

### Step 4: Checkpoint at Segment Boundaries
- At the end of each `TC[i]` interval, trigger `Manager.save()`
- Checkpointing is asynchronous (two-phase pipeline); training continues during persistence

### Step 5: Continuously Update Fit (LOF mode)
- Every `fit_interval` iterations, re-run POF on the current segment data
- If `|Δθ₁| > 0.0001` or `|Δθ₂| > 0.1`, recompute `P_e` and generate a new checkpoint schedule

### Step 6: Data Parallel Extension
At the start of each scheduling round `j`:
1. Observe `ξⱼ` from the actual loss reduction rate vs. single-worker baseline
2. Update `θ₁,j = ξⱼ · θ₁` and set `Tₛ,j = γⱼ · Tₛ`
3. Recompute `P_e,j` using Theorem 4.2
4. Apply new checkpoint schedule for round `j`

---

## 7. PyTorch Integration (Minimal Code Change)

```python
from COCI_Iterator import COCIIterator

# Replace standard training loop launcher:
COCI = COCIIterator(
    model_name='vgg19',
    dataloader=train_loader,
    ft_lambda=0.0042,        # failure rate λ (1/MTBF in minutes)
    epoch=NUM_EPOCHS,
    model=model,
    optimizer=optimizer
)

for epoch in range(start, NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        logits, probas = model(features)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        COCI.optimizer_step(loss, model, optimizer)  # replaces optimizer.step()
```

**Only two changes needed:**
1. Initialize `COCIIterator` with standard PyTorch variables
2. Replace `optimizer.step()` with `COCI.optimizer_step()`

No extra domain knowledge or tuning parameters required.

---

## 8. Key Assumptions

| Assumption | Details |
|---|---|
| Failure model | Poisson process with rate `λ` (i.e., exponentially distributed inter-failure times) |
| Checkpoint cost stability | `Tₛ` is roughly constant across training (model state size does not change) |
| Loss curve shape | Approximately exponential decay; can be accurately fit by `exp(θ₁t + θ₂)` |
| Loss normalization | Loss is normalized to [0, 1] across training duration |

---

## 9. Experimental Validation

### Hardware
- 8× NVIDIA A30 GPUs across 2 nodes (37 TB SSD, InfiniBand 10 GB/s)
- 4× NVIDIA A800 GPUs across 2 nodes

### Models Tested
| Model | Size | Checkpoint Cost |
|---|---|---|
| DenseNet201 | 235 MB | 0.13 s |
| ResNet152 | 694 MB | 0.27 s |
| VGG16_bn | 1.6 GB | 0.59 s |
| VGG19_bn | 1.7 GB | 0.60 s |
| BERT-large | 3.8 GB | 1.60 s |

### Failure Scenarios
| Scenario | λ (/min) | MTBF |
|---|---|---|
| S-1 | 0.00417 | 4 h |
| S-2 | 0.00208 | 8 h |
| S-3 | 0.00104 | 16 h |
| S-4 | 0.00024 | 70 h |

### Results Summary
| Setting | vs. CCM | vs. CheckFreq |
|---|---|---|
| Serial (best case) | −40.18% overhead | −88.54% overhead |
| Data parallel | −50.13% overhead | −63.89% overhead |
| Quality-driven scheduling | −63.89% overhead | −63.89% overhead |
| Exploratory (25% termination) | −25.00% | −66.04% |

---

## 10. Comparison with Baselines

| Property | CCM | CheckFreq | COCI |
|---|---|---|---|
| Interval type | Fixed (time-based) | Fixed (time-based) | Variable (progress-based) |
| User input required | Failure rate | Overhead threshold + failure rate | Failure rate only |
| Adapts to loss curve shape | No | No | Yes |
| Protects early stages more | No | No | Yes |
| Adapts to quality-driven scheduling | No | No | Yes |
| Performance at high failure rates | Moderate | Poor (too many ckpts) | Best |
| Performance at low failure rates | Moderate | Very poor (99.8% overhead possible) | Best |

---

## 11. Limitations and Notes

- COCI's checkpoint interval grows with training time (negative correlation with execution time). If a failure occurs **late** in training, recovery cost may be slightly higher than CCM in that specific interval.
- Benefits are most pronounced for **exploratory jobs with early termination**; for jobs that run to 100% completion, COCI and CCM perform comparably.
- `ξⱼ` (the parallelism acceleration coefficient) must be observed at the start of each scheduling round — it cannot be precomputed analytically.
- Fitting overhead is on the order of milliseconds per invocation and is negligible compared to training time.