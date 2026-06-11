from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from thesis_rrg.data.readers.mimic_reports import build_or_load_mimic_cache
from thesis_rrg.data.schemas import DatasetBundle


def _target_format(cfg: DictConfig) -> str:
    prompts = cfg.get("prompts", {})
    return str(prompts.get("target_format", "findings + impression")).strip().lower()


def _filter_non_empty(df, target_format: str):
    if target_format in {"findings", "findings only", "findings-only"}:
        return df[df["ref_findings"] != ""].copy().reset_index(drop=True)
    return df[(df["ref_findings"] != "") & (df["ref_impression"] != "")].copy().reset_index(drop=True)



def _limit(df, max_rows: int, seed: int):
    if max_rows is None or int(max_rows) == -1 or len(df) <= int(max_rows):
        return df.reset_index(drop=True)
    return df.sample(n=int(max_rows), random_state=int(seed)).reset_index(drop=True)



def build_baseline_bundle(cfg: DictConfig) -> DatasetBundle:
    df = build_or_load_mimic_cache(
        mimic_root=Path(str(cfg.data.mimic_root)),
        cache_dir=Path(str(cfg.data.cache_dir)),
        cache_tag=str(cfg.data.cache_tag),
    )
    target_format = _target_format(cfg)
    df = _filter_non_empty(df, target_format=target_format)

    train_df = _limit(df[df["split"] == "train"].copy(), int(cfg.data.max_train), int(cfg.seed))
    valid_df = _limit(df[df["split"] == "validate"].copy(), int(cfg.data.max_val), int(cfg.seed))
    test_df = _limit(df[df["split"] == "test"].copy(), int(cfg.data.max_test), int(cfg.seed))

    return DatasetBundle(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        textaug_map={},
        textaug_apply_p=0.0,
        image_aug_mode="none",
        meta={"builder": "baseline", "target_format": target_format},
    )
