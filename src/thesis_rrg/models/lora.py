from __future__ import annotations

from pathlib import Path
from typing import Any

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from thesis_rrg.models.model_utils import ADAPTER_CONFIG_NAME, has_adapter_artifact



def attach_lora(model, lora_cfg: Any):
    if not bool(lora_cfg.get("enabled", False)):
        return model

    if bool(getattr(model, "is_loaded_in_4bit", False)) or bool(getattr(model, "is_loaded_in_8bit", False)):
        model = prepare_model_for_kbit_training(model)
        
    config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 16)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        modules_to_save=list(lora_cfg.get("modules_to_save", [])),
    )

    model = get_peft_model(model, config)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    elif hasattr(model, "base_model") and hasattr(model.base_model, "enable_input_require_grads"):
        model.base_model.enable_input_require_grads()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params <= 0:
        raise RuntimeError("LoRA is enabled but no trainable parameters were found.")

    return model



def load_adapter(model, adapter_path: str, is_trainable: bool = False):
    path = _resolve_adapter_path(adapter_path)
    return PeftModel.from_pretrained(model, str(path), is_trainable=is_trainable)


def _resolve_adapter_path(adapter_path: str) -> Path:
    path = Path(adapter_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Adapter path not found: {path}")

    if has_adapter_artifact(path):
        return path

    last_path = path / "checkpoints" / "last"
    if has_adapter_artifact(last_path):
        return last_path

    best_pointer = path / "checkpoints" / "best" / "source_checkpoint.txt"
    if best_pointer.exists():
        best_path = Path(best_pointer.read_text().strip())
        if has_adapter_artifact(best_path):
            return best_path

    raise FileNotFoundError(
        f"Adapter checkpoint is missing {ADAPTER_CONFIG_NAME}: {path}. "
        "Use a PEFT adapter directory, for example checkpoints/last, or rerun training with the fixed saver."
    )
