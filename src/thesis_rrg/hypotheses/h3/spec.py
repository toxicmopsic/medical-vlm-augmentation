from __future__ import annotations

from dataclasses import dataclass


@dataclass
class H3Spec:
    name: str = "h3_e3a_safe_joint"
    mode: str = "e3a"
    augmentation_name: str = "mix"
    description: str = (
        "Safe joint augmentation hypothesis: soft image transforms + clinically equivalent "
        "LLM-paraphrased FINDINGS/IMPRESSION targets"
    )
