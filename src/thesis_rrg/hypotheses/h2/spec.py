from __future__ import annotations

from dataclasses import dataclass


@dataclass
class H2Spec:
    name: str = "h2_text_aug"
    mode: str = "e2a"
    description: str = "Text-target augmentation hypothesis (structural or LLM paraphrase)"
