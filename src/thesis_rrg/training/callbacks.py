from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
from transformers import TrainerCallback

from thesis_rrg.models.model_utils import is_main_process, is_peft_model, save_peft_adapter_artifact, unwrap_model


class TrainingProgressCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % max(int(args.logging_steps), 1) == 0 or state.global_step == 1:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                reserv = torch.cuda.memory_reserved() / (1024 ** 2)
                print(f"[Step {state.global_step}] GPU allocated={alloc:.1f}MB reserved={reserv:.1f}MB")
            else:
                print(f"[Step {state.global_step}]")
        return control

class PeftCheckpointCallback(TrainerCallback):
    def __init__(self, processor=None):
        self.processor = processor

    def on_save(self, args, state, control, model=None, **kwargs):
        trainer = kwargs.get("trainer")
        if model is None or not _is_world_process_zero(args, state, trainer):
            return control

        model = unwrap_model(model, trainer=trainer)
        if not is_peft_model(model):
            return control

        checkpoint_dir = Path(str(args.output_dir)) / f"checkpoint-{int(state.global_step)}"
        save_peft_adapter_artifact(model, self.processor, checkpoint_dir, trainer=trainer)
        _rotate_checkpoints(Path(str(args.output_dir)), int(args.save_total_limit or 0), checkpoint_dir, state)
        return control


def _is_world_process_zero(args, state, trainer=None) -> bool:
    if trainer is not None:
        return is_main_process(trainer)

    env_rank = os.environ.get("RANK")
    if env_rank is not None:
        try:
            return int(env_rank) == 0
        except ValueError:
            pass

    rank = getattr(args, "process_index", 0)
    try:
        return int(rank) == 0
    except (TypeError, ValueError):
        pass

    return bool(getattr(state, "is_world_process_zero", True))


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def _rotate_checkpoints(output_dir: Path, save_total_limit: int, current_checkpoint: Path, state) -> None:
    if save_total_limit <= 0:
        return

    checkpoints = sorted(
        [p for p in output_dir.glob("checkpoint-*") if p.is_dir() and _checkpoint_step(p) >= 0],
        key=_checkpoint_step,
    )
    if len(checkpoints) <= save_total_limit:
        return

    protected = {current_checkpoint.resolve()}
    best_checkpoint = getattr(state, "best_model_checkpoint", None)
    if best_checkpoint:
        protected.add(Path(str(best_checkpoint)).resolve())

    while len(checkpoints) > save_total_limit:
        removable = next((p for p in checkpoints if p.resolve() not in protected), None)
        if removable is None:
            break
        shutil.rmtree(removable, ignore_errors=True)
        checkpoints.remove(removable)