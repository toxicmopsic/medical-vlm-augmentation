from __future__ import annotations

from transformers import AutoProcessor



def build_processor(model_id: str, use_fast: bool = False):
    return AutoProcessor.from_pretrained(model_id, use_fast=use_fast)
