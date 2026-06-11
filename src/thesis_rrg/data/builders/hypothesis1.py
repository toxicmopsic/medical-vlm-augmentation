from __future__ import annotations

from omegaconf import DictConfig

from thesis_rrg.data.builders.baseline import build_baseline_bundle
from thesis_rrg.utils.config_utils import resolve_experiment_data_value



def build_h1_bundle(cfg: DictConfig):
    bundle = build_baseline_bundle(cfg)
    aug_name = str(resolve_experiment_data_value(cfg, "augmentation_name", default="none")).strip().lower()
    if not aug_name or aug_name == "none":
        raise ValueError(
            "H1 experiment resolved image augmentation to 'none'. "
            "Set data.augmentation_name=<photo|geo|mix|strong> or fix experiment.data.augmentation_name."
        )
    bundle.image_aug_mode = aug_name
    bundle.meta["builder"] = "h1"
    bundle.meta["image_augmentation"] = aug_name
    return bundle
