from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaselineSpec:
    name: str = "baseline_mimic"
    description: str = "No image or text augmentation; supervised on findings+impression"
