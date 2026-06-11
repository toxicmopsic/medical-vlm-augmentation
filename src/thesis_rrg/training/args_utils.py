from __future__ import annotations

from collections.abc import Callable
from typing import Any, MutableMapping

from omegaconf import DictConfig, ListConfig, OmegaConf


_EMPTY_STRINGS = {"", "none", "null"}


def _to_plain(value: Any) -> Any:
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in _EMPTY_STRINGS:
        return True
    return False


def _strip_blank_values(value: Any) -> Any:
    value = _to_plain(value)
    if isinstance(value, dict):
        return {k: _strip_blank_values(v) for k, v in value.items() if not _is_blank(v)}
    if isinstance(value, list):
        return [_strip_blank_values(v) for v in value if not _is_blank(v)]
    return value


def _to_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _add_if_supported(
    args_kwargs: MutableMapping[str, Any],
    supported_params: set[str],
    name: str,
    value: Any,
    cast: Callable[[Any], Any] | None = None,
) -> None:
    if name not in supported_params or _is_blank(value):
        return
    args_kwargs[name] = cast(value) if cast is not None else _to_plain(value)


def add_optional_training_arguments(
    args_kwargs: MutableMapping[str, Any],
    training_cfg: Any,
    supported_params: set[str],
) -> None:
    """Add version-gated Transformers/TRL TrainingArguments options."""

    _add_if_supported(args_kwargs, supported_params, "optim", training_cfg.get("optim", None), str)
    _add_if_supported(args_kwargs, supported_params, "max_grad_norm", training_cfg.get("max_grad_norm", None), float)
    _add_if_supported(args_kwargs, supported_params, "tf32", training_cfg.get("tf32", None), _to_bool)
    _add_if_supported(args_kwargs, supported_params, "torch_compile", training_cfg.get("torch_compile", None), _to_bool)
    _add_if_supported(
        args_kwargs,
        supported_params,
        "torch_compile_backend",
        training_cfg.get("torch_compile_backend", None),
        str,
    )
    _add_if_supported(
        args_kwargs,
        supported_params,
        "torch_compile_mode",
        training_cfg.get("torch_compile_mode", None),
        str,
    )
    _add_if_supported(args_kwargs, supported_params, "ddp_backend", training_cfg.get("ddp_backend", None), str)
    _add_if_supported(args_kwargs, supported_params, "ddp_timeout", training_cfg.get("ddp_timeout", None), int)
    _add_if_supported(args_kwargs, supported_params, "ddp_bucket_cap_mb", training_cfg.get("ddp_bucket_cap_mb", None), int)
    _add_if_supported(
        args_kwargs,
        supported_params,
        "ddp_find_unused_parameters",
        training_cfg.get("ddp_find_unused_parameters", None),
        _to_bool,
    )
    _add_if_supported(
        args_kwargs,
        supported_params,
        "ddp_broadcast_buffers",
        training_cfg.get("ddp_broadcast_buffers", None),
        _to_bool,
    )
    _add_if_supported(
        args_kwargs,
        supported_params,
        "ddp_static_graph",
        training_cfg.get("ddp_static_graph", None),
        _to_bool,
    )
    _add_if_supported(args_kwargs, supported_params, "save_only_model", training_cfg.get("save_only_model", None), _to_bool)
    _add_if_supported(
        args_kwargs,
        supported_params,
        "torch_empty_cache_steps",
        training_cfg.get("torch_empty_cache_steps", None),
        int,
    )

    if (
        "gradient_checkpointing_kwargs" in supported_params
        and "gradient_checkpointing_use_reentrant" in training_cfg
    ):
        args_kwargs["gradient_checkpointing_kwargs"] = {
            "use_reentrant": bool(training_cfg.get("gradient_checkpointing_use_reentrant", False))
        }

    fsdp = training_cfg.get("fsdp", None)
    if "fsdp" in supported_params and not _is_blank(fsdp):
        args_kwargs["fsdp"] = _to_plain(fsdp)

        fsdp_config = _strip_blank_values(training_cfg.get("fsdp_config", {}))
        if isinstance(fsdp_config, dict) and fsdp_config:
            if bool(fsdp_config.get("cpu_ram_efficient_loading", False)) and not bool(
                fsdp_config.get("sync_module_states", True)
            ):
                raise ValueError("training.fsdp_config.cpu_ram_efficient_loading=true requires sync_module_states=true.")
            if bool(training_cfg.get("torch_compile", False)) and not bool(fsdp_config.get("use_orig_params", True)):
                raise ValueError("torch.compile with FSDP requires training.fsdp_config.use_orig_params=true.")
            if "fsdp_config" in supported_params:
                args_kwargs["fsdp_config"] = fsdp_config
