from __future__ import annotations

from transformers import get_scheduler



def build_scheduler(optimizer, scheduler_name: str, warmup_ratio: float, total_steps: int):
    warmup_steps = int(float(warmup_ratio) * int(total_steps))
    return get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=int(total_steps),
    )
