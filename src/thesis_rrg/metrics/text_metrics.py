from __future__ import annotations

from functools import lru_cache
from typing import Any


@lru_cache
def _rouge():
    import evaluate

    return evaluate.load("rouge")


@lru_cache
def _bleu():
    import evaluate

    return evaluate.load("sacrebleu")



def compute_text_metrics(preds: list[str], refs: list[str]) -> dict[str, Any]:
    rouge = _rouge()
    bleu = _bleu()

    return {
        "n": len(preds),
        "rouge": rouge.compute(predictions=preds, references=refs, use_stemmer=True),
        "sacrebleu": bleu.compute(predictions=preds, references=[[r] for r in refs]),
    }
