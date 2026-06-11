from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image



def build_albumentations_cxr_augment(mode: str, seed: int) -> Optional[callable]:
    if mode in {"", "none", None}:
        return None

    try:
        import albumentations as A
        import cv2
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Albumentations is required for image augmentation") from e

    is_strong = mode == "strong"
    rotate_limit = 3 if not is_strong else 8
    shift_limit = 0.02 if not is_strong else 0.2
    scale_limit = 0.05 if not is_strong else 0.15

    bcl = 0.10 if not is_strong else 0.30
    gamma_limit = (90, 110) if not is_strong else (80, 120)

    std_range = (0.01, 0.02) if not is_strong else (0.04, 0.08)
    mean_range = (0.0, 0.0)
    blur_limit = (3, 5) if not is_strong else (3, 7)

    def _make_affine():
        translate_percent = {"x": (-shift_limit, shift_limit), "y": (-shift_limit, shift_limit)}
        scale = (1.0 - scale_limit, 1.0 + scale_limit)
        rotate = (-rotate_limit, rotate_limit)
        return A.Affine(
            translate_percent=translate_percent,
            scale=scale,
            rotate=rotate,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=1.0,
        )

    def _make_noise():
        return A.GaussNoise(std_range=std_range, mean_range=mean_range, p=0.5 if not is_strong else 0.9)

    geo = A.Compose([_make_affine()], seed=seed)
    photo = A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=bcl, contrast_limit=bcl, p=1.0),
            A.RandomGamma(gamma_limit=gamma_limit, p=1.0),
            _make_noise(),
            A.GaussianBlur(blur_limit=blur_limit, p=0.2 if not is_strong else 0.5),
        ],
        seed=seed,
    )

    if mode == "geo":
        aug = geo
    elif mode == "photo":
        aug = photo
    elif mode in {"mix", "strong"}:
        aug = A.Compose(geo.transforms + photo.transforms, seed=seed)
    else:
        raise ValueError("AUG_MODE must be one of: none|geo|photo|mix|strong")

    def _apply(img: Image.Image) -> Image.Image:
        arr = np.asarray(img)
        out = aug(image=arr)["image"]
        return Image.fromarray(out)

    return _apply
