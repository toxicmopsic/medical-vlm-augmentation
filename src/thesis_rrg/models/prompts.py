from __future__ import annotations

from thesis_rrg.data.preprocess.prompt_targets import build_mimic_prompt



def build_prompt(indication: str, template: str) -> str:
    return build_mimic_prompt(indication=indication, template=template)
