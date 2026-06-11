from __future__ import annotations

from typing import Any

import torch

from thesis_rrg.utils.torch_utils import move_batch_to_device



@torch.inference_mode()
def generate_batch(model, processor, prompts: list[str], images: list, generation_cfg: dict[str, Any]):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    enc = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        do_resize=False,
    )
    enc = move_batch_to_device(enc, device=device, float_dtype=dtype)

    gen_ids = model.generate(**enc, **generation_cfg)
    prompt_len = enc["input_ids"].shape[1]

    decoded = processor.batch_decode(gen_ids[:, prompt_len:], skip_special_tokens=True)
    return [str(x).strip() for x in decoded]
