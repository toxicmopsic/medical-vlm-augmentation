from __future__ import annotations

import torch



def build_optimizer(model, optim_cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=float(optim_cfg.get("lr", 1e-4)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
    )
