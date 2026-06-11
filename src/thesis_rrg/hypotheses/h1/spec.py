from __future__ import annotations

from dataclasses import dataclass


@dataclass
class H1Spec:
    name: str = "h1_image_aug"
    augmentation_name: str = "mix"
    description: str = "Image augmentation hypothesis using medically safe geometric/photometric transforms"
