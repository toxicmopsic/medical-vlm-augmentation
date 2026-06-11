from __future__ import annotations

from typing import Any, Dict, List

import torch
from transformers import AutoProcessor


def _normalize_pixel_values_shape(pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Normalize a single sample image tensor before batch stacking.

    Expected per-sample shapes for one-image inputs are typically:
    - [C, H, W]
    - [1, C, H, W]
    - [1, 1, C, H, W] (rare, extra singleton from processor path)
    """
    pv = pixel_values
    if pv.ndim == 5 and pv.shape[0] == 1:
        pv = pv.squeeze(0)
    if pv.ndim == 4 and pv.shape[0] == 1:
        pv = pv.squeeze(0)
    return pv


def _stack_pixel_values(batch: List[Dict[str, Any]]) -> torch.Tensor:
    values = [_normalize_pixel_values_shape(b["pixel_values"]) for b in batch]
    max_ndim = max(v.ndim for v in values)

    if max_ndim == 3:
        # [B, C, H, W]
        return torch.stack(values, dim=0)

    if max_ndim == 4:
        # Ensure all are [N_img, C, H, W], then stack -> [B, N_img, C, H, W].
        normalized = [v.unsqueeze(0) if v.ndim == 3 else v for v in values]
        return torch.stack(normalized, dim=0)

    raise ValueError(f"Unsupported pixel_values ndim in batch: {[v.ndim for v in values]}")


class Seq2SeqVisionLanguageCollator:
    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [b["input_ids"] for b in batch]
        attention_mask = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]

        padded = self.processor.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )
        max_len = padded["input_ids"].shape[1]
        padding_side = getattr(self.processor.tokenizer, "padding_side", "right")
        # print(f"Padding side: {padding_side}")

        padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, lab in enumerate(labels):
            if padding_side == "left":
                padded_labels[i, max_len - lab.shape[0] :] = lab
            else:
                padded_labels[i, : lab.shape[0]] = lab


        pixel_values = _stack_pixel_values(batch)

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "pixel_values": pixel_values,
            "labels": padded_labels,
        }
