from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor

from thesis_rrg.data.preprocess.image_pipeline import safe_open_report_image
from thesis_rrg.data.preprocess.prompt_targets import (
    build_full_target,
    build_mimic_prompt,
    ensure_target_terminator,
    minimal_one_line,
)
from thesis_rrg.data.transforms.structural_text import normalize_augmented_target, target_prefers_newline_after_prompt



def _sample_aug_text(study_id: str, aug_map: dict[str, list[str]], apply_p: float, rng: random.Random) -> Optional[str]:
    if apply_p <= 0.0:
        return None

    key = str(study_id)
    cands = aug_map.get(key)
    if not cands:
        return None
    if rng.random() > apply_p:
        return None

    return cands[rng.randrange(len(cands))]


class MimicReportSFTDataset(Dataset):
    def __init__(
        self,
        df_: pd.DataFrame,
        processor: AutoProcessor,
        is_train: bool,
        max_text_len: int,
        out_size: int,
        image_augment: Optional[callable],
        textaug_map: Optional[dict[str, list[str]]] = None,
        textaug_apply_p: float = 0.0,
        seed: int = 0,
        prompt_template: str = "<start_of_image> {indication} findings:",
        target_format: str = "findings + impression",
        debug_first_n: int = 0,
    ):
        self.df = df_.reset_index(drop=True)
        self.processor = processor
        self.is_train = is_train
        self.max_text_len = int(max_text_len)
        self.out_size = int(out_size)
        self.image_augment = image_augment if is_train else None
        self.textaug_map = textaug_map or {}
        self.textaug_apply_p = float(textaug_apply_p) if is_train else 0.0
        self.prompt_template = prompt_template
        self.target_format = target_format
        self.debug_first_n = int(debug_first_n)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx].to_dict()
        img = safe_open_report_image(row, out_size=self.out_size, augment=self.image_augment)

        tokenizer = self.processor.tokenizer
        prompt = build_mimic_prompt(row.get("indication", ""), template=self.prompt_template)
        # prompt_prefix = f"{prompt} "
        rng = random.Random(self.seed * 1_000_003 + int(idx))

        target_aug = _sample_aug_text(
            study_id=str(row.get("study_id", "")),
            aug_map=self.textaug_map,
            apply_p=self.textaug_apply_p,
            rng=rng,
        )
        if target_aug is not None:
            target = ensure_target_terminator(normalize_augmented_target(target_aug), tokenizer=tokenizer)
        else:
            target = build_full_target(
                row.get("ref_findings", ""),
                row.get("ref_impression", ""),
                tokenizer=tokenizer,
                target_format=self.target_format,
            )

        prompt_sep = "\n" if target_prefers_newline_after_prompt(target) else " "
        prompt_prefix = f"{prompt}{prompt_sep}"
        full_text = prompt_prefix + target

        enc = self.processor(
            text=full_text,
            images=img,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_text_len,
            do_resize=False,
        )

        input_ids = enc["input_ids"][0]
        labels = input_ids.clone()

        # Count prefix length in the same tokenization regime as `full_text`.
        full_ids_no_special = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_len,
        )["input_ids"]
        prefix_ids_no_special = tokenizer(
            prompt_prefix,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_text_len,
        )["input_ids"]

        # Handle special token prefix (e.g., BOS) that appears in `enc["input_ids"]`.
        special_prefix_len = max(0, len(input_ids) - len(full_ids_no_special))
        prefix_len = min(len(input_ids), special_prefix_len + len(prefix_ids_no_special))
        labels[:prefix_len] = -100

        sample = {
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "pixel_values": enc["pixel_values"][0],
            "labels": labels,
        }

        if idx < self.debug_first_n:
            sample["debug_prompt"] = prompt
            sample["debug_target"] = target

        return sample


class MimicEvalDataset(Dataset):
    def __init__(self, df_: pd.DataFrame):
        self.df = df_.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx].to_dict()
        return {
            "study_id": str(row["study_id"]),
            "dcm_path": str(row["dcm_path"]),
            "indication": row.get("indication", ""),
            "ref_findings": row.get("ref_findings", ""),
            "ref_impression": row.get("ref_impression", ""),
        }
