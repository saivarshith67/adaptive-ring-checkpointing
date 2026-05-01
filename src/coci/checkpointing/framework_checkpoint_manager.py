"""Checkpoint managers for framework-native checkpointing backends."""

import os
import time
from typing import Optional

import torch
import torch.distributed as dist

from .checkpoint_manager import CheckpointManager


FRAMEWORK_BACKENDS = {
    "pytorch-lightning",
    "hf-trainer",
    "deepspeed",
    "fsdp",
    "wandb-artifacts",
}


class FrameworkCheckpointManager(CheckpointManager):
    """Checkpoint manager that delegates saves to framework-native APIs when possible.

    The public interface intentionally matches ``CheckpointManager`` so the training
    pipeline can swap backend strategies without changing metrics or recovery flow.
    If a framework object is not supplied, the manager falls back to the canonical
    torch checkpoint format used by the rest of this project.
    """

    def __init__(
        self,
        backend: str,
        checkpoint_dir: str = "checkpoints",
        is_ddp_wrapped: bool = False,
        lightning_trainer=None,
        hf_trainer=None,
        deepspeed_engine=None,
        wandb_run=None,
        artifact_type: str = "model-checkpoint",
    ):
        if backend not in FRAMEWORK_BACKENDS:
            raise ValueError(f"Unsupported framework checkpoint backend: {backend}")
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            is_ddp_wrapped=is_ddp_wrapped,
        )
        self.backend = backend
        self.lightning_trainer = lightning_trainer
        self.hf_trainer = hf_trainer
        self.deepspeed_engine = deepspeed_engine
        self.wandb_run = wandb_run
        self.artifact_type = artifact_type
        self.framework_save_count = 0
        self.framework_load_count = 0
        self.artifact_log_count = 0
        self.last_framework_metadata = None

    def _backend_dir(self, epoch: int, checkpoint_id: Optional[str] = None) -> str:
        label = checkpoint_id if checkpoint_id is not None else f"epoch_{epoch}"
        return os.path.join(self.checkpoint_dir, f"{self.backend}_{label}")

    def _rank(self) -> int:
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    def _barrier(self) -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def _record_framework_save(
        self,
        path: str,
        epoch: int,
        checkpoint_id: Optional[str],
        metric: Optional[float],
        loss: float,
        duration: float,
        delegated: bool,
    ) -> None:
        self.num_checkpoints += 1
        self.total_checkpoint_time += duration
        self.framework_save_count += 1
        self.last_save_metadata = {
            "path": path,
            "epoch": epoch,
            "checkpoint_id": checkpoint_id,
            "metric": metric,
            "loss": loss,
            "duration_sec": duration,
            "load_source": self.backend,
            "checkpoint_backend": self.backend,
            "framework_delegated": delegated,
        }
        self.last_framework_metadata = self.last_save_metadata

    def _save_torch_compatible(
        self,
        model,
        optimizer,
        epoch: int,
        loss: float,
        metric: Optional[float],
        checkpoint_id: Optional[str],
    ) -> str:
        path = self._resolve_checkpoint_path(epoch, checkpoint_id=checkpoint_id)
        state_dict = (
            model.module.state_dict()
            if self.is_ddp_wrapped and hasattr(model, "module")
            else model.state_dict()
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict()
                if optimizer is not None
                else None,
                "loss": loss,
                "metric": metric,
                "checkpoint_id": checkpoint_id,
                "checkpoint_backend": self.backend,
                "save_time": time.time(),
            },
            path,
        )
        return path

    def _save_lightning(
        self,
        model,
        optimizer,
        epoch: int,
        loss: float,
        metric: Optional[float],
        checkpoint_id: Optional[str],
    ) -> tuple[str, bool]:
        path = self._resolve_checkpoint_path(epoch, checkpoint_id=checkpoint_id)
        if self.lightning_trainer is not None and hasattr(
            self.lightning_trainer, "save_checkpoint"
        ):
            self.lightning_trainer.save_checkpoint(path)
            return path, True
        return self._save_torch_compatible(
            model, optimizer, epoch, loss, metric, checkpoint_id
        ), False

    def _save_hf_trainer(
        self,
        model,
        optimizer,
        epoch: int,
        loss: float,
        metric: Optional[float],
        checkpoint_id: Optional[str],
    ) -> tuple[str, bool]:
        output_dir = self._backend_dir(epoch, checkpoint_id)
        if self.hf_trainer is not None and hasattr(self.hf_trainer, "save_model"):
            os.makedirs(output_dir, exist_ok=True)
            self.hf_trainer.save_model(output_dir)
            if hasattr(self.hf_trainer, "save_state"):
                self.hf_trainer.save_state()
            metadata_path = os.path.join(output_dir, "coci_checkpoint_meta.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "metric": metric,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_backend": self.backend,
                },
                metadata_path,
            )
            return output_dir, True
        return self._save_torch_compatible(
            model, optimizer, epoch, loss, metric, checkpoint_id
        ), False

    def _save_deepspeed(
        self,
        model,
        optimizer,
        epoch: int,
        loss: float,
        metric: Optional[float],
        checkpoint_id: Optional[str],
    ) -> tuple[str, bool]:
        engine = self.deepspeed_engine or (
            model if hasattr(model, "save_checkpoint") else None
        )
        tag = checkpoint_id if checkpoint_id is not None else f"epoch_{epoch}"
        if engine is not None and hasattr(engine, "save_checkpoint"):
            client_state = {
                "epoch": epoch,
                "loss": loss,
                "metric": metric,
                "checkpoint_id": checkpoint_id,
                "checkpoint_backend": self.backend,
            }
            engine.save_checkpoint(
                self.checkpoint_dir,
                tag=tag,
                client_state=client_state,
            )
            return os.path.join(self.checkpoint_dir, tag), True
        return self._save_torch_compatible(
            model, optimizer, epoch, loss, metric, checkpoint_id
        ), False

    def _save_fsdp(
        self,
        model,
        optimizer,
        epoch: int,
        loss: float,
        metric: Optional[float],
        checkpoint_id: Optional[str],
    ) -> tuple[str, bool]:
        path = self._resolve_checkpoint_path(epoch, checkpoint_id=checkpoint_id)
        state_dict = None
        delegated = False
        try:
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            if isinstance(model, FSDP):
                cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                    state_dict = model.state_dict()
                delegated = True
        except Exception:
            state_dict = None

        if state_dict is None:
            try:
                from fairscale.nn.data_parallel import FullyShardedDataParallel as FairScaleFSDP

                if isinstance(model, FairScaleFSDP):
                    state_dict = model.state_dict()
                    delegated = True
            except Exception:
                state_dict = None

        if state_dict is None:
            state_dict = (
                model.module.state_dict()
                if self.is_ddp_wrapped and hasattr(model, "module")
                else model.state_dict()
            )

        if self._rank() == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict()
                    if optimizer is not None
                    else None,
                    "loss": loss,
                    "metric": metric,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_backend": self.backend,
                    "fsdp_full_state_dict": delegated,
                    "save_time": time.time(),
                },
                path,
            )
        return path, delegated

    def _log_wandb_artifact(
        self,
        path: str,
        epoch: int,
        checkpoint_id: Optional[str],
        metric: Optional[float],
    ) -> bool:
        run = self.wandb_run
        wandb_module = None
        if run is None:
            try:
                import wandb as wandb_module

                run = getattr(wandb_module, "run", None)
            except Exception:
                run = None

        if run is None:
            return False

        try:
            if wandb_module is None:
                import wandb as wandb_module

            label = checkpoint_id if checkpoint_id is not None else f"epoch_{epoch}"
            artifact = wandb_module.Artifact(
                name=f"checkpoint-{label}",
                type=self.artifact_type,
                metadata={
                    "epoch": epoch,
                    "checkpoint_id": checkpoint_id,
                    "metric": metric,
                    "checkpoint_backend": self.backend,
                },
            )
            if os.path.isdir(path):
                artifact.add_dir(path)
            else:
                artifact.add_file(path)
            run.log_artifact(artifact)
            self.artifact_log_count += 1
            return True
        except Exception:
            return False

    def save(
        self,
        model,
        optimizer,
        epoch,
        loss,
        metric=None,
        checkpoint_id=None,
    ):
        rank = self._rank()
        if rank != 0 and self.backend not in {"deepspeed", "fsdp"}:
            self._barrier()
            return

        start = time.time()
        delegated = False

        if self.backend == "pytorch-lightning":
            path, delegated = self._save_lightning(
                model, optimizer, epoch, loss, metric, checkpoint_id
            )
        elif self.backend == "hf-trainer":
            path, delegated = self._save_hf_trainer(
                model, optimizer, epoch, loss, metric, checkpoint_id
            )
        elif self.backend == "deepspeed":
            path, delegated = self._save_deepspeed(
                model, optimizer, epoch, loss, metric, checkpoint_id
            )
        elif self.backend == "fsdp":
            path, delegated = self._save_fsdp(
                model, optimizer, epoch, loss, metric, checkpoint_id
            )
        elif self.backend == "wandb-artifacts":
            path = self._save_torch_compatible(
                model, optimizer, epoch, loss, metric, checkpoint_id
            )
            delegated = self._log_wandb_artifact(
                path, epoch, checkpoint_id, metric
            )
        else:
            raise ValueError(f"Unsupported framework checkpoint backend: {self.backend}")

        duration = time.time() - start
        if rank == 0:
            self._record_framework_save(
                path=path,
                epoch=epoch,
                checkpoint_id=checkpoint_id,
                metric=metric,
                loss=loss,
                duration=duration,
                delegated=delegated,
            )
            label = checkpoint_id if checkpoint_id is not None else f"epoch_{epoch}"
            print(
                f"[Checkpoint:{self.backend}] Saved {label} | "
                f"delegated={delegated} | Time: {duration:.4f}s"
            )

        self._barrier()

    def _copy_hf_model_if_needed(self, path: str) -> Optional[str]:
        meta_path = os.path.join(path, "coci_checkpoint_meta.pt")
        if os.path.exists(meta_path):
            return meta_path
        return None

    def load_latest(self, model, optimizer, device, checkpoint_injector=None):
        engine = self.deepspeed_engine or (
            model if hasattr(model, "load_checkpoint") else None
        )
        if self.backend == "deepspeed" and engine is not None:
            entries = [
                os.path.join(self.checkpoint_dir, item)
                for item in os.listdir(self.checkpoint_dir)
            ] if os.path.exists(self.checkpoint_dir) else []
            dirs = [entry for entry in entries if os.path.isdir(entry)]
            if dirs:
                latest_dir = max(dirs, key=os.path.getmtime)
                tag = os.path.basename(latest_dir)
                _, client_state = engine.load_checkpoint(
                    self.checkpoint_dir,
                    tag=tag,
                )
                client_state = client_state or {}
                epoch = int(client_state.get("epoch", 0))
                best_metric = client_state.get("metric", 0.0)
                self.framework_load_count += 1
                self.last_load_metadata = {
                    "path": latest_dir,
                    "epoch": epoch,
                    "checkpoint_id": client_state.get("checkpoint_id"),
                    "metric": best_metric,
                    "load_source": self.backend,
                    "checkpoint_backend": self.backend,
                }
                # epoch from checkpoint is 1-indexed (epochs completed)
                return epoch, best_metric

        result = super().load_latest(
            model,
            optimizer,
            device,
            checkpoint_injector=checkpoint_injector,
        )
        self.framework_load_count += 1 if self.last_load_metadata else 0
        if self.last_load_metadata:
            self.last_load_metadata["checkpoint_backend"] = self.backend
            self.last_load_metadata["load_source"] = self.backend
        return result

    def get_runtime_metrics(self):
        metrics = super().get_runtime_metrics()
        metrics.update(
            {
                "load_source": self.backend,
                "checkpoint_backend": self.backend,
                "framework_save_count": self.framework_save_count,
                "framework_load_count": self.framework_load_count,
                "artifact_log_count": self.artifact_log_count,
                "last_checkpoint_path": (
                    self.last_save_metadata.get("path")
                    if self.last_save_metadata
                    else None
                ),
                "framework_delegated": (
                    self.last_save_metadata.get("framework_delegated", False)
                    if self.last_save_metadata
                    else False
                ),
            }
        )
        return metrics


def create_checkpoint_manager(
    backend: str,
    checkpoint_dir: str = "checkpoints",
    is_ddp_wrapped: bool = False,
    **kwargs,
):
    """Create the right checkpoint manager for a backend name."""
    if backend in {"normal", "epoch", "convergence"}:
        return CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            is_ddp_wrapped=is_ddp_wrapped,
        )
    if backend in FRAMEWORK_BACKENDS:
        return FrameworkCheckpointManager(
            backend=backend,
            checkpoint_dir=checkpoint_dir,
            is_ddp_wrapped=is_ddp_wrapped,
            **kwargs,
        )
    raise ValueError(f"Unknown checkpoint backend: {backend}")
