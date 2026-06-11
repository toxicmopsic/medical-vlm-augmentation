from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SampleMeta:
    study_id: str
    split: str
