from __future__ import annotations

import os
from pathlib import Path

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


def is_main_process(trainer) -> bool:
    rank = os.environ.get("RANK")
    if rank is not None:
        try:
            return int(rank) == 0
        except ValueError:
            pass

    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "is_main_process"):
        return bool(accelerator.is_main_process)

    if hasattr(trainer, "is_world_process_zero"):
        return bool(trainer.is_world_process_zero())

    return True


def wait_for_everyone(trainer) -> None:
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None and hasattr(accelerator, "wait_for_everyone"):
        accelerator.wait_for_everyone()


def unwrap_model(model, trainer=None):
    if trainer is not None:
        accelerator = getattr(trainer, "accelerator", None)
        if accelerator is not None and hasattr(accelerator, "unwrap_model"):
            try:
                model = accelerator.unwrap_model(model)
            except Exception:
                pass

    while hasattr(model, "module"):
        model = model.module

    return model


def is_peft_model(model) -> bool:
    return hasattr(model, "peft_config")


def has_adapter_artifact(path: str | Path) -> bool:
    root = Path(path)
    return (root / ADAPTER_CONFIG_NAME).exists() and any((root / name).exists() for name in ADAPTER_WEIGHT_NAMES)


def validate_adapter_artifact(path: str | Path) -> None:
    root = Path(path)
    if not (root / ADAPTER_CONFIG_NAME).exists():
        raise FileNotFoundError(f"Missing {ADAPTER_CONFIG_NAME} in adapter checkpoint: {root}")
    if not any((root / name).exists() for name in ADAPTER_WEIGHT_NAMES):
        expected = ", ".join(ADAPTER_WEIGHT_NAMES)
        raise FileNotFoundError(f"Missing adapter weights ({expected}) in adapter checkpoint: {root}")


def save_peft_adapter_artifact(model, processor, out_dir: str | Path, trainer=None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = unwrap_model(model, trainer=trainer)
    if not is_peft_model(model):
        raise TypeError(f"Expected a PEFT model with peft_config, got {type(model).__name__}.")

    model.save_pretrained(str(out), safe_serialization=True)
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(out))

    validate_adapter_artifact(out)
    return out


def save_model_artifact(model, processor, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(out))
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(out))

    return out


def save_trainer_model_artifact(trainer, processor, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(out))

    wait_for_everyone(trainer)
    if is_main_process(trainer):
        model = unwrap_model(trainer.model, trainer=trainer)
        if is_peft_model(model):
            save_peft_adapter_artifact(model, processor, out, trainer=trainer)
        else:
            trainer.save_model(str(out))
            if processor is not None and hasattr(processor, "save_pretrained"):
                processor.save_pretrained(str(out))
    wait_for_everyone(trainer)

    return out