# Soft Error Injection Mechanism — Implementation Spec Sheet

**Based on:** Rojas, Pérez & Meneses (2024), *"A characterization of soft-error sensitivity in data-parallel and model-parallel distributed deep learning"*, Journal of Parallel and Distributed Computing 190, 104879.

---

## 1. Overview

This document specifies the implementation of a checkpoint-based fault injection system for studying silent data corruption (SDC) in distributed deep learning training. The mechanism injects bit-flips into HDF5 checkpoint files to emulate soft errors without requiring specialized hardware.

---

## 2. System Architecture

The injector is composed of three pipeline stages:

```
┌─────────────────────────┐     ┌──────────────────────┐     ┌─────────────────────────┐
│  Network Feature        │────▶│  Training Process    │────▶│  Fault Injection        │
│  Extractor              │     │  Analyzer            │     │  Engine                 │
│                         │     │                      │     │                         │
│  • Layer types          │     │  • GPU/CPU count     │     │  • Fault type           │
│  • Layer count          │     │  • DT mechanism      │     │  • Fault location       │
│  • FP precision         │     │    (DDP/HVD/GPipe)   │     │  • Bit range            │
│  • Weights per layer    │     │  • Process mapping   │     │  • Fault probability    │
│  • NNQ quantification   │     │                      │     │  • Auto precision       │
└─────────────────────────┘     └──────────────────────┘     └─────────────────────────┘
```

---

## 3. Prerequisites

### 3.1 Software Stack

| Component        | Version / Notes                         |
|------------------|-----------------------------------------|
| OS               | Ubuntu 20.04.3 LTS (Linux 5.4.0)        |
| Python           | 3.8+                                    |
| PyTorch          | 1.9.0+                                  |
| H5PY (HDF5)      | 3.1.0+                                  |
| Horovod          | 0.24.2+ (for HVD experiments)           |
| CUDA             | 11.4+                                   |
| NVIDIA APEX      | Latest (for mixed-precision experiments) |

### 3.2 Hardware Requirements

- Minimum 2 GPUs (8 recommended for full fidelity)
- NVIDIA A100 or equivalent (Tensor Core support recommended)
- Sufficient RAM to hold full model checkpoints per process

---

## 4. Core Concepts

### 4.1 Checkpoint Alteration Strategy

Rather than injecting errors at runtime, this mechanism corrupts saved checkpoint files before they are reloaded into training. This approach:

- Requires no instrumentation of the running DL framework
- Is framework-agnostic (applicable to DDP, HVD, GPipe, and others)
- Supports deterministic reproduction of experiments
- Operates on HDF5-format checkpoint files

### 4.2 Structures Subject to Injection

Two in-memory structures are targeted independently:

**Neural Network Model** — the weight tensors of each layer stored in the checkpoint's `state_dict`.

**Optimizer State** — gradient momentum/variance buffers stored separately by PyTorch (e.g., SGD momentum, Adam first/second moments).

> These must be injected separately to isolate their respective sensitivities.

### 4.3 Floating-Point Bit Layout (IEEE 754)

Bit-flip impact varies sharply by position:

| Precision | Sign bits | Exponent bits | Mantissa bits | Critical bit (MSB of exponent) |
|-----------|-----------|---------------|---------------|-------------------------------|
| FP16      | 1         | 5             | 10            | Bit 14                        |
| FP32      | 1         | 8             | 23            | Bit 30                        |
| FP64      | 1         | 11            | 52            | Bit 62                        |

Flipping the most significant exponent bit (critical bit) produces extreme values (EV), NaN, or INF — the most catastrophic outcomes. Mantissa and sign-bit flips are generally recoverable.

---

## 5. Injector Configuration Parameters

| Parameter         | Type         | Description                                                                 |
|-------------------|--------------|-----------------------------------------------------------------------------|
| `fault_type`      | enum         | `RANDOM_BIT`, `SPECIFIC_BIT`, `SIGN_BIT`, `EXPONENT_MSB`                   |
| `fault_location`  | string/int   | Target structure: `"model"`, `"optimizer"`, or specific layer index         |
| `bit_range`       | tuple(int)   | Inclusive range of bits eligible for flipping, e.g. `(0, 31)` for FP32     |
| `fault_probability` | float      | Probability of injecting a fault per eligible value (0.0–1.0)              |
| `num_bit_flips`   | int          | Total number of bit-flips to inject per checkpoint file                     |
| `num_processes`   | int          | Number of distributed processes that load a corrupted checkpoint            |
| `total_processes` | int          | Total number of processes in the distributed run                            |
| `auto_precision`  | bool         | Automatically adapt bit positions based on detected FP precision             |
| `dt_mechanism`    | enum         | `DDP`, `HVD`, `GPIPE`                                                       |
| `target_layers`   | list[int]    | Specific layer indices to corrupt (leave empty for random selection)         |
| `log_injection`   | bool         | Write injection log for equivalent injection across mechanisms               |
| `load_log`        | string/path  | Path to a prior injection log (for equivalent injection)                    |

---

## 6. Injection Workflow

### Step 1 — Network Feature Extraction

```
inspect_checkpoint(checkpoint_path):
    → enumerate layers (name, type, shape)
    → detect floating-point precision (FP16 / FP32 / FP64)
    → count weights per layer
    → identify non-model structures (BN buffers, optimizer state, etc.)
    → output: NNQ (Neural Network Quantification) metadata
```

### Step 2 — Process and Hardware Mapping

```
map_processes(dt_mechanism, num_gpus):
    → DDP:   one checkpoint file per rank; model replicated across all ranks
    → HVD:   one checkpoint file per rank; ring-allreduce synchronization
    → GPipe: one checkpoint file per rank; each rank owns distinct layer subset
    → select P processes (out of N total) to receive corrupted checkpoints
```

### Step 3 — Bit-Flip Injection

```
inject_bitflips(checkpoint_file, config):
    → open HDF5 file in read/write mode
    → if fault_location == "model":  target state_dict weight tensors
    → if fault_location == "optimizer": target optimizer state buffers
    → for each selected injection site:
        → read raw bytes of the floating-point value
        → XOR the target bit(s) with 1
        → write corrupted bytes back
    → if log_injection: write log {layer, bit_position, bit_value_before, bit_value_after}
    → save corrupted checkpoint to output path
```

### Step 4 — Equivalent Injection (Cross-Mechanism Fairness)

To fairly compare DDP vs HVD (which use slightly different checkpoint formats):

```
equivalent_inject(source_log, target_checkpoint, target_mechanism):
    → load injection log from source mechanism (e.g., DDP)
    → parse: injected_layer, bit_flip_position, fp_bit_index
    → locate equivalent weight in target_checkpoint (e.g., HVD format)
    → apply identical bit-flip
    → applies to both model and optimizer structures
```

> Note: Equivalent injection between GPipe and DDP/HVD is not feasible because GPipe reorganizes the network into a sequential pipeline, changing the layer structure.

### Step 5 — Training Restart and Evaluation

```
resume_training(corrupted_checkpoint, clean_checkpoint, config):
    → load corrupted checkpoint into designated process(es)
    → load clean checkpoint into all remaining processes
    → run training for specified epochs after restart
    → record:
        → final accuracy
        → NaN occurrence (training collapse)
        → accuracy delta vs. base (flag if Δ > 5%)
        → AccR (accuracy reduction count)
        → TF (training fail without NaN count)
```

---

## 7. Determinism Requirements

ANN training is inherently stochastic. To isolate the effect of injected faults, determinism must be enforced:

```python
import torch, random, numpy as np, os

def enforce_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
```

All processes in a distributed run must use the same seed and configuration.

---

## 8. Experiment Configurations

### 8.1 Single Bit-Flip Sensitivity

| Parameter          | Value                   |
|--------------------|-------------------------|
| `num_bit_flips`    | 1                       |
| `num_processes`    | 1                       |
| `total_processes`  | 8                       |
| `fault_location`   | `"model"` or `"optimizer"` (separate runs) |
| `fault_type`       | `RANDOM_BIT`            |
| `num_trainings`    | 180 per mechanism       |
| Evaluation epoch   | 1 epoch post-restart    |

### 8.2 Multiple Bit-Flip Sensitivity

| `num_bit_flips` | `num_processes` | Notes                          |
|-----------------|-----------------|--------------------------------|
| 1               | 4 or 8          | —                              |
| 10              | 1, 4, or 8      | —                              |
| 100             | 1, 4, or 8      | Collapse expected at 8/8       |
| 500, 1000       | —               | Produces 100% NaN; omit        |

### 8.3 Floating-Point Precision Sensitivity

Target only the critical MSB of the exponent per precision level:

| Precision | APEX Mode | Target Bit |
|-----------|-----------|------------|
| FP16      | O3        | Bit 14     |
| FP32      | O1        | Bit 30     |
| FP64      | Simulated (cast FP32 value to FP64) | Bit 62 |

### 8.4 SDC Layer Propagation

| Parameter              | Value                           |
|------------------------|---------------------------------|
| Network                | ResNet18 (18 layers)            |
| `num_bit_flips`        | 1000 (1 process) or 125 (8 processes) |
| `target_layers`        | Layer 1 (first), Layer 8 (middle) |
| Exclude                | Bit 30 (to prevent NaN)         |
| Training epochs        | 1 or 10 after restart           |
| Evaluation             | Per-layer relative error vs. clean checkpoint |

Relative error per layer:

```
RE_i = (A2_i - A1_i) / |A1_i|

where:
  A1 = weight array from unaltered checkpoint
  A2 = weight array from corrupted checkpoint
  i  = weight index within the layer
```

---

## 9. Evaluation Metrics

| Metric   | Definition                                                                 |
|----------|----------------------------------------------------------------------------|
| RWC      | Restart Without Change — training accuracy unchanged after corrupt reload  |
| NaN      | Training collapse — model produced Not-a-Number values                     |
| AccR     | Accuracy Reduction — final accuracy drops >5% vs. base                     |
| TF       | Training Fail — incorrect accuracy output, no NaN generated                |
| ACC      | Mean accuracy across all non-NaN runs                                      |
| STD      | Standard deviation of accuracy                                             |
| CV       | Coefficient of variation (STD / ACC)                                       |

**Base accuracy thresholds (CIFAR100, ResNet):**

| Mechanism      | Structure  | Base Accuracy |
|----------------|------------|---------------|
| DDP            | Model      | ~71%          |
| HVD            | Model      | ~69%          |
| GPipe          | Model      | ~81%          |
| DDP            | Optimizer  | ~70%          |
| HVD            | Optimizer  | ~69%          |
| GPipe          | Optimizer  | ~82%          |
| DDP (FP16)     | Model      | ~60%          |
| HVD (FP16)     | Model      | ~59%          |
| GPipe (FP16)   | Model      | ~79%          |

---

## 10. Dataset and Model Configuration

| Component      | Value                                      |
|----------------|--------------------------------------------|
| Dataset        | CIFAR100 (60,000 images, 32×32 px)        |
| Train split    | 50,000 images                              |
| Test split     | 10,000 images                              |
| Model          | ResNet50 (primary), ResNet18 (propagation) |
| Optimizer      | SGD                                        |
| LR Scheduler   | ReduceLROnPlateau                          |
| Training epochs| Up to 100; checkpoint loaded at epoch 20  |
| Restart epoch  | 20                                         |

---

## 11. Known Limitations and Considerations

- Equivalent injection between GPipe and DDP/HVD cannot be performed due to structural differences in layer organization under pipeline parallelism.
- Sign-bit flips do not produce distinct training behavior due to ReLU zeroing negative activations — sign-bit targeted experiments can be omitted.
- Bit-flips that cause no observable effect may have landed on non-computational structures within the checkpoint (e.g., metadata, unused buffers).
- At very high bit-flip rates (500+), 100% NaN collapse should be expected; these runs should be excluded from accuracy analysis.
- FP64 experiments require a cast-based simulation (FP32 value cast to FP64 before injection) since PyTorch defaults to FP32 operations.
- Results between GPipe and DDP/HVD are indicative but not directly comparable due to the absence of equivalent injection.

---

## 12. Output Artifacts

| Artifact                        | Format   | Description                                        |
|---------------------------------|----------|----------------------------------------------------|
| Corrupted checkpoint file       | HDF5     | Modified checkpoint with injected bit-flips        |
| Injection log                   | JSON/CSV | Records layer, bit position, before/after values   |
| Training results                | CSV      | ACC, STD, CV, NaN, AccR, TF per configuration      |
| Layer relative error arrays     | NumPy    | Per-layer RE values for propagation analysis       |
| Boxplot figures                 | PNG/SVG  | Visual SDC propagation across layers               |

---

*Spec version 1.0 — April 2026*