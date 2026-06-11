from __future__ import annotations

from typing import Optional

from thesis_rrg.data.transforms.albumentations_aug import build_albumentations_cxr_augment



def build_cxr_augment(mode: str, seed: int) -> Optional[callable]:
    return build_albumentations_cxr_augment(mode=mode, seed=seed)
