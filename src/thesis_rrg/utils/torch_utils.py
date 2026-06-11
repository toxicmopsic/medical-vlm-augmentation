from __future__ import annotations

from typing import Optional

import torch



def resolve_torch_dtype(name: str | None) -> Optional[torch.dtype]:
    if name is None:
        return None

    key = str(name).strip().lower()
    if key in {"", "none"}:
        return None
    if key == "auto":
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"Unsupported torch dtype: {name}")



def move_batch_to_device(batch: dict, device: torch.device, float_dtype: torch.dtype | None = None) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if float_dtype is not None and v.is_floating_point():
                v = v.to(dtype=float_dtype)
        out[k] = v
    return out
