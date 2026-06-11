from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf



def resolve_path(path_like: str) -> Path:
    return Path(path_like).expanduser().resolve()



def cfg_get(cfg: DictConfig, key: str, default=None):
    return cfg.get(key, default)



def to_float(cfg: DictConfig, key: str, default: float) -> float:
    raw = cfg.get(key, default)
    return float(raw)



def to_int(cfg: DictConfig, key: str, default: int) -> int:
    raw = cfg.get(key, default)
    return int(raw)

def _is_missing_value(value: Any, sentinels: Iterable[Any]) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized == "" or normalized in {str(x).strip().lower() for x in sentinels if x is not None}
    return value in sentinels


def resolve_experiment_data_value(
    cfg: DictConfig,
    key: str,
    *,
    default: Any = None,
    sentinels: Iterable[Any] = ("", "none", None),
) -> Any:
    """
    Resolve data options that may be declared either under `data.*` or
    `experiment.data.*`.

    Hydra composes `configs/experiment/*.yaml` under the `experiment` node in
    this project. Builders, however, consume top-level `cfg.data.*`. This helper
    lets explicit top-level overrides win, while still honoring experiment-local
    defaults when the top-level value is only the common baseline sentinel.
    """
    top_value = default
    if "data" in cfg and key in cfg.data:
        top_value = cfg.data.get(key)

    if not _is_missing_value(top_value, sentinels):
        return top_value

    exp_value = default
    if "experiment" in cfg and "data" in cfg.experiment and key in cfg.experiment.data:
        exp_value = cfg.experiment.data.get(key)

    if not _is_missing_value(exp_value, sentinels):
        return exp_value

    return top_value


def dataset_bundle_summary(bundle) -> dict[str, Any]:
    return {
        "meta": dict(getattr(bundle, "meta", {}) or {}),
        "image_aug_mode": str(getattr(bundle, "image_aug_mode", "none")),
        "textaug_apply_p": float(getattr(bundle, "textaug_apply_p", 0.0)),
        "textaug_map_size": int(len(getattr(bundle, "textaug_map", {}) or {})),
        "n_train": int(len(getattr(bundle, "train_df", []))),
        "n_valid": int(len(getattr(bundle, "valid_df", []))),
        "n_test": int(len(getattr(bundle, "test_df", []))),
    }


def config_to_plain_dict(cfg: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(cfg, resolve=True)
    return value if isinstance(value, dict) else {}
