from __future__ import annotations

import random
import re
from collections import Counter
from typing import Iterable

HIDDEN_CONTROL_RE = re.compile(r"[\u200b-\u200d\ufeff]")
WS_TABS_RE = re.compile(r"[ \t]+")
MANY_NL_RE = re.compile(r"\n{3,}")
COMPACT_NL_RE = re.compile(r"\n{2,}")
MULTISPACE_RE = re.compile(r"\s+")

BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", flags=re.M)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")
DEPENDENCY_CUES_RE = re.compile(
    r"\b(compared to|comparison|previous|prior|interval|again|as before|therefore|however|this|these|above)\b",
    flags=re.I,
)


def norm_keep_newlines(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    t = HIDDEN_CONTROL_RE.sub("", t)
    t = WS_TABS_RE.sub(" ", t)
    t = MANY_NL_RE.sub("\n\n", t)
    return t.strip()


def norm_one_line(text: str | None) -> str:
    return MULTISPACE_RE.sub(" ", norm_keep_newlines(text)).strip()


def is_bulleted_list(text: str | None) -> bool:
    return bool(BULLET_RE.search(norm_keep_newlines(text)))


def strip_bullet_prefix(line: str) -> str:
    return BULLET_RE.sub("", line, count=1).strip()


def bullet_items(text: str | None) -> list[str]:
    items = []
    for line in norm_keep_newlines(text).splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(strip_bullet_prefix(line) if BULLET_RE.match(line) else line)
    return [x for x in items if x]


def split_sentences(text: str | None) -> list[str]:
    t = norm_keep_newlines(text)
    if not t:
        return []
    if is_bulleted_list(t):
        return bullet_items(t)
    parts = [norm_keep_newlines(x) for x in SENT_SPLIT_RE.split(t)]
    return [x for x in parts if x] or [t]


def _rstrip_sentence_period(text: str) -> str:
    return text.strip().rstrip(".").strip()


def _ensure_sentence_period(text: str) -> str:
    t = text.strip()
    if not t:
        return ""
    return t if t[-1] in ".!?" else f"{t}."


def join_sentences(sentences: Iterable[str], style: int = 0) -> str:
    sents = [norm_keep_newlines(s) for s in sentences if norm_keep_newlines(s)]
    if not sents:
        return ""
    if style == 1:
        return " ".join(_ensure_sentence_period(_rstrip_sentence_period(s)) for s in sents)
    if style == 2:
        return "; ".join(_rstrip_sentence_period(s) for s in sents) + "."
    if style == 3:
        return "\n".join(f"- {_rstrip_sentence_period(s)}" for s in sents)
    return " ".join(sents)


def paragraph_to_bullets(text: str | None, marker: str = "-") -> str:
    sents = split_sentences(text)
    if len(sents) < 2:
        return norm_keep_newlines(text)
    marker = marker if marker in {"-", "*", "\u2022"} else "-"
    return "\n".join(f"{marker} {_rstrip_sentence_period(s)}" for s in sents)


def bullets_to_paragraph(text: str | None) -> str:
    items = bullet_items(text)
    if not items:
        return norm_keep_newlines(text)
    return join_sentences(items, style=1)


def can_shuffle(sentences: Iterable[str]) -> bool:
    sents = list(sentences)
    if len(sents) < 2:
        return False
    return not any(DEPENDENCY_CUES_RE.search(s) for s in sents)


def merge_short_adjacent(sentences: list[str], min_len_chars: int = 35) -> list[str]:
    out = []
    i = 0
    while i < len(sentences):
        cur = sentences[i]
        if i + 1 < len(sentences) and (len(cur) < min_len_chars or len(sentences[i + 1]) < min_len_chars):
            out.append(norm_keep_newlines(f"{_rstrip_sentence_period(cur)}; {sentences[i + 1].lstrip()}"))
            i += 2
        else:
            out.append(cur)
            i += 1
    return out


def shuffle_sentences(text: str | None, rng: random.Random, style: int = 1) -> str:
    sents = split_sentences(text)
    if not can_shuffle(sents):
        return norm_keep_newlines(text)
    shuffled = sents[:]
    rng.shuffle(shuffled)
    return norm_keep_newlines(join_sentences(shuffled, style=style))


def structural_aug_once(text: str | None, rng: random.Random) -> str:
    text0 = norm_keep_newlines(text)
    if not text0:
        return text0

    ops = ["shuffle", "bullets_to_para", "para_to_bullets", "punct"]
    rng.shuffle(ops)
    sents = split_sentences(text0)

    for op in ops:
        if op == "shuffle" and can_shuffle(sents):
            s2 = sents[:]
            rng.shuffle(s2)
            if rng.random() < 0.35:
                s2 = merge_short_adjacent(s2, min_len_chars=35)
            out = norm_keep_newlines(join_sentences(s2, style=rng.randint(0, 2)))
            if out and out != text0:
                return out

        if op == "bullets_to_para" and is_bulleted_list(text0):
            out = norm_keep_newlines(bullets_to_paragraph(text0))
            if out and out != text0:
                return out

        if op == "para_to_bullets" and len(sents) >= 2 and not is_bulleted_list(text0):
            out = norm_keep_newlines(paragraph_to_bullets(text0))
            if out and out != text0:
                return out

        if op == "punct":
            out = text0.replace(" .", ".").replace(" ;", ";")
            out = re.sub(r"\s+,", ",", out)
            out = re.sub(r"\s+\.", ".", out)
            out = norm_keep_newlines(out)
            if out and out != text0:
                return out

    return text0


def sentence_signature(text: str | None) -> Counter[str]:
    parts = split_sentences(text)
    normalized = []
    for part in parts:
        t = strip_bullet_prefix(part)
        t = _rstrip_sentence_period(t)
        t = re.sub(r"[;:]+$", "", t).strip().lower()
        t = MULTISPACE_RE.sub(" ", t)
        if t:
            normalized.append(t)
    return Counter(normalized)


def has_sentence_loss_or_duplication(src: str | None, cand: str | None) -> bool:
    return sentence_signature(src) != sentence_signature(cand)


def normalize_augmented_target(text: str | None) -> str:
    return COMPACT_NL_RE.sub("\n", norm_keep_newlines(text))


def target_prefers_newline_after_prompt(target: str | None) -> bool:
    t = normalize_augmented_target(target)
    return "\n" in t or bool(re.match(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", t))


def format_findings_impression_target(findings: str | None, impression: str | None) -> str:
    f = normalize_augmented_target(findings)
    i = normalize_augmented_target(impression)
    if not f and not i:
        return ""
    if not i:
        return f
    if not f:
        return f"impression:\n{i}" if target_prefers_newline_after_prompt(i) else f"impression: {i}"

    impression_block = f"impression:\n{i}" if target_prefers_newline_after_prompt(i) else f"impression: {i}"
    sep = "\n" if target_prefers_newline_after_prompt(f) else " "
    return f"{f}{sep}{impression_block}".strip()
