from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

from torch.utils.data import Dataset

from thesis_rrg.data.preprocess.image_pipeline import safe_open_report_image
from thesis_rrg.data.preprocess.prompt_targets import build_full_target, build_mimic_prompt, ensure_target_terminator
from thesis_rrg.data.transforms.structural_text import normalize_augmented_target

def _sample_aug_text(study_id: str, aug_map: dict[str, list[str]], apply_p: float, rng: random.Random) -> str | None:
    if apply_p <= 0.0:
        return None
    cands = aug_map.get(str(study_id))
    if not cands or rng.random() > apply_p:
        return None
    return cands[rng.randrange(len(cands))]

class ChatDataset(Dataset):
    """TRL-ready dataset where each example has a `messages` field."""

    def __init__(self, rows: list[Dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def chat_prompt_text(prompt: str) -> str:
    return prompt.replace("<start_of_image>", "").strip()


def build_user_chat(prompt: str, image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": chat_prompt_text(prompt)},
            ],
        }
    ]


class MimicReportChatDataset(Dataset):
    """Lazy MedGemma/Gemma-chat dataset for TRL SFT."""

    def __init__(
        self,
        df_,
        tokenizer,
        prompt_template: str,
        target_format: str,
        out_size: int = 896,
        is_train: bool = False,
        textaug_map: dict[str, list[str]] | None = None,
        textaug_apply_p: float = 0.0,
        seed: int = 0,
    ):
        self.df = df_.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.target_format = target_format
        self.out_size = int(out_size)
        self.is_train = bool(is_train)
        self.textaug_map = textaug_map or {}
        self.textaug_apply_p = float(textaug_apply_p) if is_train else 0.0
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx].to_dict()
        image = safe_open_report_image(row, out_size=self.out_size)
        prompt = build_mimic_prompt(row.get("indication", ""), template=self.prompt_template)
        target_aug = _sample_aug_text(
            study_id=str(row.get("study_id", "")),
            aug_map=self.textaug_map,
            apply_p=self.textaug_apply_p,
            rng=random.Random(self.seed * 1_000_003 + int(idx)),
        )
        if target_aug is not None:
            target = ensure_target_terminator(normalize_augmented_target(target_aug), tokenizer=self.tokenizer)
        else:
            target = build_full_target(
                row.get("ref_findings", ""),
                row.get("ref_impression", ""),
                tokenizer=self.tokenizer,
                target_format=self.target_format,
            )
        messages = build_user_chat(prompt, image)
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target}],
            }
        )
        return {"messages": messages}
