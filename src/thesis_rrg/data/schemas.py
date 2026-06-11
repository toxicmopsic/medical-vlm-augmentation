from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


@dataclass
class DataPaths:
    mimic_root: Path
    cache_dir: Path
    e2a_dir: Path
    e2c_merged_path: Path


@dataclass
class DataPreparationConfig:
    cache_tag: str = "v1"
    batch_size: int = 1
    num_workers: int = 4
    max_text_len: int = 768
    debug_first_n: int = 0
    max_train: int = -1
    max_val: int = 2000
    max_test: int = -1


@dataclass
class DatasetBundle:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame
    textaug_map: Dict[str, list[str]] = field(default_factory=dict)
    textaug_apply_p: float = 0.0
    image_aug_mode: str = "none"
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrainInput:
    bundle: DatasetBundle
    tokenizer_max_len: int
    batch_size: int
    num_workers: int
    seed: int
    model_id: str
    max_text_len: int
