from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from thesis_rrg.data.builders.baseline import build_baseline_bundle
from thesis_rrg.data.transforms.text_augment import load_e2a_map, load_e2c_map
from thesis_rrg.utils.config_utils import resolve_experiment_data_value



def build_h2_bundle(cfg: DictConfig):
    bundle = build_baseline_bundle(cfg)

    mode = str(resolve_experiment_data_value(cfg, "experiment_mode", default="none")).strip().lower()
    if mode == "e2a":
        path = Path(str(cfg.data.e2a_cache_path))
        aug_map = load_e2a_map(path)
        apply_p = float(cfg.data.e2a_apply_p)
    elif mode == "e2c":
        path = Path(str(cfg.data.e2c_merged_path))
        aug_map = load_e2c_map(path)
        apply_p = float(cfg.data.e2c_apply_p)
    elif mode in {"none", "baseline"}:
        raise ValueError(
            "H2 experiment resolved text augmentation mode to 'none'. "
            "Set data.experiment_mode=<e2a|e2c> or fix experiment.data.experiment_mode."
        )
    else:
        raise ValueError(f"Unsupported h2 experiment_mode={mode}")

    if not aug_map:
        raise ValueError(
            f"H2 experiment_mode={mode} resolved an empty text augmentation map from {path}. "
            "Check that the parquet exists, has study_id/ex_id and cand_full/aug_full columns, "
            "and that data.e2c_merged_path/data.e2a_cache_path points to the intended file."
        )

    bundle.textaug_map = aug_map
    bundle.textaug_apply_p = apply_p
    bundle.meta["builder"] = "h2"
    bundle.meta["h2_mode"] = mode
    bundle.meta["text_augmentation_path"] = str(path)
    bundle.meta["text_augmentation_size"] = str(len(aug_map))
    return bundle
