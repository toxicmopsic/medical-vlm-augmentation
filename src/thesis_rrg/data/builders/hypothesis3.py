from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from thesis_rrg.data.builders.baseline import build_baseline_bundle
from thesis_rrg.data.transforms.text_augment import load_e2c_map
from thesis_rrg.utils.config_utils import resolve_experiment_data_value


_SAFE_IMAGE_MODES = {"none", "geo", "photo", "mix"}


def _resolve_safe_image_mode(aug_name: str) -> str:
    mode = str(aug_name).strip().lower()
    if mode not in _SAFE_IMAGE_MODES:
        safe = ", ".join(sorted(_SAFE_IMAGE_MODES))
        raise ValueError(
            f"Unsupported H3 image augmentation '{aug_name}'. "
            f"E3a allows only clinically safe modes: {safe}."
        )
    return mode


def build_h3_bundle(cfg: DictConfig):
    bundle = build_baseline_bundle(cfg)

    mode = str(resolve_experiment_data_value(cfg, "h3_mode", default="e3a")).strip().lower()
    if mode in {"none", ""}:
        mode = str(resolve_experiment_data_value(cfg, "experiment_mode", default="e3a")).strip().lower()
    if mode != "e3a":
        raise ValueError(f"Unsupported h3 mode={mode}. Supported: e3a")

    image_mode = _resolve_safe_image_mode(str(resolve_experiment_data_value(cfg, "augmentation_name", default="mix")))
    e2c_path = Path(str(cfg.data.e2c_merged_path))
    text_map = load_e2c_map(e2c_path)
    apply_p = float(cfg.data.e2c_apply_p)
    require_textaug = bool(cfg.data.get("h3_require_textaug", True))
    if require_textaug and not text_map:
        raise ValueError(
            f"H3/E3a requires non-empty E2C paraphrase targets at {e2c_path}. "
            "Build/merge E2C artifacts first."
        )

    bundle.image_aug_mode = image_mode
    bundle.textaug_map = text_map
    bundle.textaug_apply_p = apply_p if text_map else 0.0

    bundle.meta["builder"] = "h3"
    bundle.meta["h3_mode"] = mode
    bundle.meta["image_augmentation"] = image_mode
    bundle.meta["text_augmentation"] = "e2c_llm_paraphrase_filtered"
    return bundle
