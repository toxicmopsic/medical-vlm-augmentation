from __future__ import annotations

import os
from typing import Any

import torch
from transformers import AutoModelForImageTextToText

from thesis_rrg.models.quantization import build_quant_config
from thesis_rrg.utils.torch_utils import resolve_torch_dtype


def _is_fsdp_enabled(training_cfg: Any) -> bool:
    fsdp = training_cfg.get("fsdp", None)
    if fsdp is None:
        return False
    if isinstance(fsdp, str):
        return fsdp.strip().lower() not in {"", "none", "null"}
    return bool(fsdp)

def _configure_fsdp_loading(training_cfg: Any) -> None:
    fsdp_config = training_cfg.get("fsdp_config", {}) or {}
    if not bool(fsdp_config.get("cpu_ram_efficient_loading", False)):
        return
    if not bool(fsdp_config.get("sync_module_states", True)):
        raise ValueError("FSDP CPU-RAM efficient loading requires training.fsdp_config.sync_module_states=true.")

    try:
        from accelerate.utils import enable_fsdp_ram_efficient_loading

        enable_fsdp_ram_efficient_loading()
    except Exception:
        os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "true"

def load_base_model(model_cfg: Any, training_cfg: Any):
    if _is_fsdp_enabled(training_cfg):
        _configure_fsdp_loading(training_cfg)

    torch_dtype = resolve_torch_dtype(str(model_cfg.get("torch_dtype", "auto")))
    quant_config = build_quant_config(model_cfg.get("quantization", {}))

    kwargs = {
        "torch_dtype": torch_dtype,
    }

    device_map = training_cfg.get("device_map", None)
    if device_map not in (None, "", "none"):
        if _is_fsdp_enabled(training_cfg):
            raise ValueError("Do not set training.device_map with FSDP. Use torchrun and CUDA_VISIBLE_DEVICES instead.")
        kwargs["device_map"] = device_map

    if quant_config is not None:
        kwargs["quantization_config"] = quant_config

    attn_impl = model_cfg.get("attn_implementation", "")
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    model = AutoModelForImageTextToText.from_pretrained(str(model_cfg.model_id), **kwargs)

    if bool(training_cfg.get("gradient_checkpointing", False)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    return model
