import pandas as pd
import pytest
from omegaconf import OmegaConf

from thesis_rrg.data.builders import hypothesis1 as h1_builder
from thesis_rrg.data.builders import hypothesis2 as h2_builder
from thesis_rrg.data.schemas import DatasetBundle
from thesis_rrg.utils.config_utils import resolve_experiment_data_value


def _empty_bundle() -> DatasetBundle:
    empty = pd.DataFrame()
    return DatasetBundle(
        train_df=empty.copy(),
        valid_df=empty.copy(),
        test_df=empty.copy(),
        textaug_map={},
        textaug_apply_p=0.0,
        image_aug_mode="none",
        meta={"builder": "baseline"},
    )


def test_experiment_data_value_falls_back_to_experiment_node():
    cfg = OmegaConf.create(
        {
            "data": {"augmentation_name": "none"},
            "experiment": {"data": {"augmentation_name": "photo"}},
        }
    )

    assert resolve_experiment_data_value(cfg, "augmentation_name", default="none") == "photo"


def test_top_level_data_override_wins():
    cfg = OmegaConf.create(
        {
            "data": {"augmentation_name": "geo"},
            "experiment": {"data": {"augmentation_name": "photo"}},
        }
    )

    assert resolve_experiment_data_value(cfg, "augmentation_name", default="none") == "geo"


def test_h1_builder_uses_experiment_data_when_top_level_is_none(monkeypatch):
    monkeypatch.setattr(h1_builder, "build_baseline_bundle", lambda cfg: _empty_bundle())
    cfg = OmegaConf.create(
        {
            "data": {"augmentation_name": "none"},
            "experiment": {
                "family": "h1",
                "data": {"augmentation_name": "photo"},
            },
        }
    )

    bundle = h1_builder.build_h1_bundle(cfg)

    assert bundle.image_aug_mode == "photo"
    assert bundle.meta["image_augmentation"] == "photo"


def test_h1_builder_rejects_noop_augmentation(monkeypatch):
    monkeypatch.setattr(h1_builder, "build_baseline_bundle", lambda cfg: _empty_bundle())
    cfg = OmegaConf.create({"data": {"augmentation_name": "none"}, "experiment": {"family": "h1"}})

    with pytest.raises(ValueError, match="resolved image augmentation to 'none'"):
        h1_builder.build_h1_bundle(cfg)


def test_h2_builder_uses_experiment_data_when_top_level_is_none(monkeypatch):
    monkeypatch.setattr(h2_builder, "build_baseline_bundle", lambda cfg: _empty_bundle())
    monkeypatch.setattr(h2_builder, "load_e2c_map", lambda path: {"123": ["No edema impression: Normal."]})

    cfg = OmegaConf.create(
        {
            "data": {
                "experiment_mode": "none",
                "e2c_merged_path": "/tmp/e2c.parquet",
                "e2c_apply_p": 1.0,
                "e2a_cache_path": "/tmp/e2a.parquet",
                "e2a_apply_p": 1.0,
            },
            "experiment": {
                "family": "h2",
                "data": {"experiment_mode": "e2c"},
            },
        }
    )

    bundle = h2_builder.build_h2_bundle(cfg)

    assert bundle.meta["h2_mode"] == "e2c"
    assert bundle.textaug_apply_p == 1.0
    assert len(bundle.textaug_map) == 1


def test_h2_builder_rejects_empty_textaug(monkeypatch):
    monkeypatch.setattr(h2_builder, "build_baseline_bundle", lambda cfg: _empty_bundle())
    monkeypatch.setattr(h2_builder, "load_e2c_map", lambda path: {})

    cfg = OmegaConf.create(
        {
            "data": {
                "experiment_mode": "none",
                "e2c_merged_path": "/tmp/e2c.parquet",
                "e2c_apply_p": 1.0,
            },
            "experiment": {
                "family": "h2",
                "data": {"experiment_mode": "e2c"},
            },
        }
    )

    with pytest.raises(ValueError, match="empty text augmentation map"):
        h2_builder.build_h2_bundle(cfg)
