from __future__ import annotations

import torch



def mask_token_ids(labels: torch.Tensor, token_ids: list[int], mask_value: int = -100) -> torch.Tensor:
    out = labels.clone()
    for token_id in token_ids:
        out[out == token_id] = mask_value
    return out
