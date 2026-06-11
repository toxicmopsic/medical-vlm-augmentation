from __future__ import annotations

from typing import Optional

import torch
from transformers import BitsAndBytesConfig

from thesis_rrg.utils.torch_utils import resolve_torch_dtype



def build_quant_config(cfg) -> Optional[BitsAndBytesConfig]:
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return None

    compute_dtype = resolve_torch_dtype(cfg.get("bnb_4bit_compute_dtype", "bf16")) or torch.bfloat16
    quant_storage = resolve_torch_dtype(cfg.get("bnb_4bit_quant_storage", "bf16")) or torch.bfloat16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
        bnb_4bit_quant_type=str(cfg.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_storage=quant_storage,
    )
