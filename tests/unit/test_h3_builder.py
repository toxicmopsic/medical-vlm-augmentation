import pandas as pd
import pytest
from omegaconf import OmegaConf

from thesis_rrg.data.builders import hypothesis3 as h3_builder
from thesis_rrg.data.schemas import DatasetBundle


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


def test_h3_e3a_build_composes_image_and_text_aug(monkeypatch):
    monkeypatch.setattr(h3_builder, "build_baseline_bundle", lambda cfg: _empty_bundle())
    monkeypatch.setattr(
        h3_builder,
        "load_e2c_map",
        lambda _path: {"1001": ["FINDINGS: clear lungs\nIMPRESSION: no acute disease"]},
    )

    cfg = OmegaConf.create(
        {
            "data": {
                "h3_mode": "e3a",
                "augmentation_name": "mix",
                "e2c_apply_p": 1.0,
                "e2c_merged_path": "/tmp/e2c_best.parquet",
            }
        }
    )
    bundle = h3_builder.build_h3_bundle(cfg)

    assert bundle.image_aug_mode == "mix"
    assert bundle.textaug_apply_p == 1.0
    assert "1001" in bundle.textaug_map
    assert bundle.meta["builder"] == "h3"
    assert bundle.meta["h3_mode"] == "e3a"
    assert bundle.meta["text_augmentation"] == "e2c_llm_paraphrase_filtered"


def test_h3_e3a_rejects_non_safe_image_mode(monkeypatch):
    monkeypatch.setattr(h3_builder, "build_baseline_bundle", lambda cfg: _empty_bundle())

    cfg = OmegaConf.create(
        {
            "data": {
                "h3_mode": "e3a",
                "augmentation_name": "strong",
                "e2c_apply_p": 1.0,
                "e2c_merged_path": "/tmp/e2c_best.parquet",
            }
        }
    )

    with pytest.raises(ValueError, match="clinically safe modes"):
        h3_builder.build_h3_bundle(cfg)


def test_h3_e3a_requires_non_empty_text_aug_by_default(monkeypatch):
    monkeypatch.setattr(h3_builder, "build_baseline_bundle", lambda cfg: _empty_bundle())
    monkeypatch.setattr(h3_builder, "load_e2c_map", lambda _path: {})

    cfg = OmegaConf.create(
        {
            "data": {
                "h3_mode": "e3a",
                "augmentation_name": "mix",
                "e2c_apply_p": 1.0,
                "e2c_merged_path": "/tmp/e2c_best.parquet",
            }
        }
    )

    with pytest.raises(ValueError, match="requires non-empty E2C paraphrase targets"):
        h3_builder.build_h3_bundle(cfg)
