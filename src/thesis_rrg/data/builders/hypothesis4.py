from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from thesis_rrg.data.builders.baseline import build_baseline_bundle
from thesis_rrg.hypotheses.h4.synthetic import h4_manifest_paths
from thesis_rrg.utils.config_utils import resolve_experiment_data_value


_STAGE_TO_MANIFEST = {
    "real_baseline": "train_real_baseline",
    "b0_real_only": "train_real_baseline",
    "synthetic_pretrain": "train_synthetic_pretrain",
    "synth_pretrain": "train_synthetic_pretrain",
    "real_finetune": "train_real_finetune",
    "real_ft": "train_real_finetune",
}


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"H4 training manifest is missing: {path}. "
            "Build it first with `python -m thesis_rrg.cli.prepare_data experiment=h4_synth_pretrain_real_ft data.prepare_mode=h4_all`."
        )
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"H4 training manifest is empty: {path}") from exc
    if df.empty:
        raise ValueError(f"H4 training manifest is empty: {path}")
    return df


def _limit_manifest(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows is None or int(max_rows) == -1 or len(df) <= int(max_rows):
        return df.reset_index(drop=True)
    return df.sample(n=int(max_rows), random_state=int(seed)).reset_index(drop=True)


def build_h4_bundle(cfg: DictConfig):
    bundle = build_baseline_bundle(cfg)

    stage = str(resolve_experiment_data_value(cfg, "h4_train_stage", default="real_baseline")).strip().lower()
    manifest_key = _STAGE_TO_MANIFEST.get(stage)
    if manifest_key is None:
        supported = ", ".join(sorted(_STAGE_TO_MANIFEST))
        raise ValueError(f"Unsupported h4_train_stage={stage!r}. Supported: {supported}")

    paths = h4_manifest_paths(cfg)
    manifest_path = paths[manifest_key]
    train_df = _load_manifest(manifest_path)
    train_df = _limit_manifest(train_df, int(cfg.data.max_train), int(cfg.seed))

    bundle.train_df = train_df
    bundle.textaug_map = {}
    bundle.textaug_apply_p = 0.0
    bundle.image_aug_mode = "none"
    bundle.meta["builder"] = "h4"
    bundle.meta["h4_train_stage"] = stage
    bundle.meta["h4_train_manifest"] = str(manifest_path)
    bundle.meta["h4_train_rows"] = str(len(train_df))
    bundle.meta["h4_artifact_name"] = str(paths.get("artifact_name", ""))
    bundle.meta["h4_artifact_dir"] = str(paths.get("artifact_dir", ""))
    return bundle
