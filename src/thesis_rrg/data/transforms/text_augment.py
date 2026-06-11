from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from thesis_rrg.data.transforms.structural_text import (
    format_findings_impression_target,
    normalize_augmented_target,
    norm_one_line
)


def _to_key(x) -> str:
    try:
        return str(int(x))
    except Exception:
        return str(x)



def _clean_text_keep_newlines(s) -> str:
    if s is None:
        return ""
    return normalize_augmented_target(str(s))

def _clean_text_one_line(s) -> str:
    if s is None:
        return ""
    return norm_one_line(str(s))

def _build_map(
    df: pd.DataFrame,
    key_col: str,
    text_col: str,
    *,
    keep_newlines: bool = False,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    cleaner = _clean_text_keep_newlines if keep_newlines else _clean_text_one_line

    for key, group in df.groupby(key_col):
        seen = set()
        vals = []
        for value in group[text_col].tolist():
            text = cleaner(value)
            if text and text not in seen:
                seen.add(text)
                vals.append(text)
        if vals:
            out[_to_key(key)] = vals
    return out



def load_e2a_map(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}

    df = pd.read_parquet(path)
    if len(df) == 0 or "study_id" not in df.columns:
        return {}
    if {"aug_findings", "aug_impression"}.issubset(df.columns):
        df = df.copy()
        df["_target_full"] = [
            format_findings_impression_target(findings, impression)
            for findings, impression in zip(df["aug_findings"], df["aug_impression"])
        ]
        return _build_map(df, key_col="study_id", text_col="_target_full", keep_newlines=True)

    if "aug_full" not in df.columns:
        return {}
    return _build_map(df, key_col="study_id", text_col="aug_full", keep_newlines=True)


def load_e2c_map(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}

    df = pd.read_parquet(path)
    if len(df) == 0:
        return {}

    key_col = "study_id" if "study_id" in df.columns else ("ex_id" if "ex_id" in df.columns else None)
    text_col = "cand_full" if "cand_full" in df.columns else ("aug_full" if "aug_full" in df.columns else None)

    if key_col is None or text_col is None:
        raise ValueError(
            f"E2c parquet at {path} must contain key column [study_id|ex_id] and text column [cand_full|aug_full]."
        )

    return _build_map(df, key_col=key_col, text_col=text_col, keep_newlines=False)
