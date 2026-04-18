import json
import math
import os
import random
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


class FaultType(str, Enum):
    RANDOM_BIT = "RANDOM_BIT"
    SPECIFIC_BIT = "SPECIFIC_BIT"
    SIGN_BIT = "SIGN_BIT"
    EXPONENT_MSB = "EXPONENT_MSB"


@dataclass
class CheckpointFaultConfig:
    enabled: bool = True
    fault_type: FaultType = FaultType.RANDOM_BIT
    fault_location: str = "model"  # model | optimizer | all
    bit_range: Optional[Tuple[int, int]] = None
    fault_probability: float = 1.0
    num_bit_flips: int = 1
    num_processes: int = 1
    total_processes: Optional[int] = None
    auto_precision: bool = True
    target_layers: List[int] = field(default_factory=list)
    specific_bit_position: Optional[int] = None
    log_injection: bool = True
    injection_log_path: Optional[str] = None
    load_log: Optional[str] = None
    seed: int = 42


def enforce_determinism(seed: int = 42) -> None:
    """Force deterministic behavior to isolate bit-flip impact from training noise."""
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class CheckpointBitFlipInjector:
    """Inject bit-flips into checkpoint tensors before model/optimizer restore."""

    _DTYPE_META = {
        torch.float16: {"bits": 16, "pack": "<e", "critical": 14},
        torch.float32: {"bits": 32, "pack": "<f", "critical": 30},
        torch.float64: {"bits": 64, "pack": "<d", "critical": 62},
    }

    def __init__(
        self,
        config: CheckpointFaultConfig,
        rank: int = 0,
        world_size: int = 1,
        dt_mechanism: str = "DDP",
    ):
        self.config = config
        self.rank = rank
        self.world_size = max(1, world_size)
        self.dt_mechanism = dt_mechanism
        self.total_injections = 0

    def inject(self, checkpoint: Dict, checkpoint_path: str) -> Dict:
        if not self.config.enabled:
            return checkpoint

        selected_ranks = self._select_target_ranks(checkpoint_path)
        if self.rank not in selected_ranks:
            return checkpoint

        if self.config.load_log:
            entries = self._load_entries_from_log(self.config.load_log)
            if not entries:
                return checkpoint
            applied_entries = self._apply_logged_injections(checkpoint, entries)
            if applied_entries and self.config.log_injection:
                self._write_log(applied_entries)
            return checkpoint

        candidates = self._collect_candidates(checkpoint)
        if not candidates:
            return checkpoint

        rng = self._rng_for_checkpoint(checkpoint_path)
        planned_flips = max(0, int(self.config.num_bit_flips))
        max_attempts = max(16, planned_flips * 20)
        attempts = 0
        applied = 0
        entries: List[Dict] = []

        while applied < planned_flips and attempts < max_attempts:
            attempts += 1
            if rng.random() > float(self.config.fault_probability):
                continue

            tensor_path, tensor, layer_idx = rng.choice(candidates)
            flat = tensor.view(-1)
            if flat.numel() == 0:
                continue

            value_index = rng.randrange(flat.numel())
            bit_position = self._choose_bit_position(tensor.dtype, rng)
            if bit_position is None:
                continue

            before_value = float(flat[value_index].item())
            after_value, before_bit, after_bit = self._flip_value_bit(
                before_value,
                tensor.dtype,
                bit_position,
            )
            flat[value_index] = after_value

            entry = {
                "timestamp": time.time(),
                "rank": self.rank,
                "world_size": self.world_size,
                "dt_mechanism": self.dt_mechanism,
                "checkpoint": os.path.basename(checkpoint_path),
                "fault_type": self.config.fault_type.value,
                "fault_location": self.config.fault_location,
                "tensor_path": tensor_path,
                "layer_index": layer_idx,
                "flat_index": int(value_index),
                "bit_position": int(bit_position),
                "bit_value_before": int(before_bit),
                "bit_value_after": int(after_bit),
                "value_before": before_value,
                "value_after": float(after_value),
                "dtype": str(tensor.dtype),
            }
            entries.append(entry)
            applied += 1

        self.total_injections += applied
        if entries and self.config.log_injection:
            self._write_log(entries)

        return checkpoint

    def _collect_candidates(self, checkpoint: Dict) -> List[Tuple[str, torch.Tensor, Optional[int]]]:
        location = self.config.fault_location.lower().strip()
        include_model = location in {"model", "all"} or location.isdigit()
        include_optimizer = location in {"optimizer", "all"}

        candidates: List[Tuple[str, torch.Tensor, Optional[int]]] = []

        model_state = checkpoint.get("model_state_dict", {})
        model_items = [
            (name, tensor)
            for name, tensor in model_state.items()
            if isinstance(tensor, torch.Tensor) and tensor.dtype in self._DTYPE_META
        ]

        if location.isdigit():
            self.config.target_layers = [int(location)]

        if include_model and model_items:
            target_layers = set(self.config.target_layers)
            for idx, (name, tensor) in enumerate(model_items):
                if target_layers and idx not in target_layers:
                    continue
                candidates.append((f"model.{name}", tensor, idx))

        if include_optimizer:
            optimizer_state = checkpoint.get("optimizer_state_dict", {})
            state = optimizer_state.get("state", {})
            for param_key, param_state in state.items():
                if not isinstance(param_state, dict):
                    continue
                for state_key, value in param_state.items():
                    if isinstance(value, torch.Tensor) and value.dtype in self._DTYPE_META:
                        path = f"optimizer.state.{param_key}.{state_key}"
                        candidates.append((path, value, None))

        return candidates

    def _select_target_ranks(self, checkpoint_path: str) -> Sequence[int]:
        total_processes = self.config.total_processes or self.world_size
        total_processes = max(1, int(total_processes))
        num_processes = max(1, min(int(self.config.num_processes), total_processes))

        if num_processes >= total_processes:
            return list(range(total_processes))

        rng = self._rng_for_checkpoint(checkpoint_path)
        ranks = list(range(total_processes))
        return sorted(rng.sample(ranks, k=num_processes))

    def _rng_for_checkpoint(self, checkpoint_path: str) -> random.Random:
        checkpoint_key = os.path.basename(checkpoint_path)
        seed_material = f"{self.config.seed}:{checkpoint_key}:{self.dt_mechanism}"
        seed = abs(hash(seed_material)) % (2**32)
        return random.Random(seed)

    def _choose_bit_position(
        self,
        dtype: torch.dtype,
        rng: random.Random,
    ) -> Optional[int]:
        meta = self._DTYPE_META.get(dtype)
        if meta is None:
            return None

        if self.config.fault_type == FaultType.SIGN_BIT:
            return meta["bits"] - 1

        if self.config.fault_type == FaultType.EXPONENT_MSB:
            return int(meta["critical"])

        if self.config.fault_type == FaultType.SPECIFIC_BIT:
            if self.config.specific_bit_position is not None:
                bit = int(self.config.specific_bit_position)
            else:
                bit = int(meta["critical"])
            return bit if 0 <= bit < meta["bits"] else None

        if self.config.bit_range is None or not self.config.auto_precision:
            low = 0
            high = meta["bits"] - 1
        else:
            low = max(0, int(self.config.bit_range[0]))
            high = min(meta["bits"] - 1, int(self.config.bit_range[1]))

        if low > high:
            return None
        return rng.randint(low, high)

    def _flip_value_bit(
        self,
        value: float,
        dtype: torch.dtype,
        bit_position: int,
    ) -> Tuple[float, int, int]:
        meta = self._DTYPE_META[dtype]
        pack_fmt = meta["pack"]
        byte_length = meta["bits"] // 8

        packed = struct.pack(pack_fmt, float(value))
        bitset = int.from_bytes(packed, byteorder="little", signed=False)
        before_bit = (bitset >> bit_position) & 1
        flipped = bitset ^ (1 << bit_position)
        after_bit = (flipped >> bit_position) & 1

        unpacked = struct.unpack(
            pack_fmt,
            int(flipped).to_bytes(byte_length, byteorder="little", signed=False),
        )[0]
        return float(unpacked), int(before_bit), int(after_bit)

    def _apply_logged_injections(self, checkpoint: Dict, entries: List[Dict]) -> List[Dict]:
        candidates = {path: (tensor, layer_idx) for path, tensor, layer_idx in self._collect_candidates(checkpoint)}
        applied_entries: List[Dict] = []

        for entry in entries:
            path = entry.get("tensor_path")
            flat_index = entry.get("flat_index")
            bit_position = entry.get("bit_position")

            if path not in candidates:
                continue
            tensor, layer_idx = candidates[path]
            flat = tensor.view(-1)
            if flat.numel() == 0:
                continue

            idx = int(flat_index)
            if idx < 0 or idx >= flat.numel():
                continue

            bit = int(bit_position)
            before_value = float(flat[idx].item())
            after_value, before_bit, after_bit = self._flip_value_bit(
                before_value,
                tensor.dtype,
                bit,
            )
            flat[idx] = after_value

            applied_entries.append(
                {
                    "timestamp": time.time(),
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "dt_mechanism": self.dt_mechanism,
                    "checkpoint": entry.get("checkpoint"),
                    "fault_type": entry.get("fault_type", "REPLAY"),
                    "fault_location": entry.get("fault_location", self.config.fault_location),
                    "tensor_path": path,
                    "layer_index": layer_idx,
                    "flat_index": idx,
                    "bit_position": bit,
                    "bit_value_before": before_bit,
                    "bit_value_after": after_bit,
                    "value_before": before_value,
                    "value_after": float(after_value),
                    "dtype": str(tensor.dtype),
                    "replayed_from_log": self.config.load_log,
                }
            )

        self.total_injections += len(applied_entries)
        return applied_entries

    def _load_entries_from_log(self, log_path: str) -> List[Dict]:
        if not os.path.exists(log_path):
            return []

        entries: List[Dict] = []
        with open(log_path, "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
            if not raw:
                return []
            if raw[0] == "[":
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    entries.extend([e for e in parsed if isinstance(e, dict)])
            else:
                for line in raw.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if isinstance(item, dict):
                            entries.append(item)
                    except json.JSONDecodeError:
                        continue

        return entries

    def _write_log(self, entries: Iterable[Dict]) -> None:
        path = self.config.injection_log_path
        if not path:
            os.makedirs("checkpoints/injection_logs", exist_ok=True)
            path = os.path.join(
                "checkpoints/injection_logs",
                f"injection_rank{self.rank}_{int(time.time())}.jsonl",
            )
        else:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

        with open(path, "a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry) + "\n")
