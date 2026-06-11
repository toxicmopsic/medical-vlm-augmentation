from __future__ import annotations

from thesis_rrg.data.preprocess.normalization import minimal_normalize
from thesis_rrg.data.transforms.structural_text import norm_keep_newlines



def build_mimic_prompt(indication: str, template: str = "<start_of_image> {indication} findings:") -> str:
    ind = minimal_normalize(indication)
    if not ind:
        return template.format(indication="").replace("  ", " ").strip()
    return template.format(indication=ind).strip()



def minimal_one_line(text: str | None) -> str:
    return minimal_normalize(text)

def minimal_keep_newlines(text: str | None) -> str:
    return norm_keep_newlines(text)

def ensure_target_terminator(body: str, tokenizer) -> str:
    body = minimal_keep_newlines(body)

    eos_token = getattr(tokenizer, "eos_token", None) if tokenizer is not None else None
    if eos_token is not None:
        if body.endswith(eos_token):
            return body
        return body + eos_token

    if body.endswith(" END_REPORT"):
        return body
    return body + " END_REPORT"


def _append_target_terminator(body: str, tokenizer) -> str:
    # Keep EOS handling centralized so findings-only and full-report targets stop identically
    # Main option: explicit EOS token in training text
    if tokenizer is not None and getattr(tokenizer, "eos_token", None) is not None:
        return body + tokenizer.eos_token
    # Fallback: explicit textual end marker
    return body + " END_REPORT"


def _normalize_target_format(target_format: str | None) -> str:
    fmt = (target_format or "findings + impression").strip().lower().replace("_", " ")
    if fmt in {"findings", "findings only", "findings-only"}:
        return "findings"
    if fmt in {"findings + impression", "findings and impression", "full", "full report"}:
        return "findings + impression"
    raise ValueError(f"Unsupported target_format={target_format!r}")

def build_target_text(
    findings: str,
    impression: str,
    tokenizer,
    target_format: str | None = "findings + impression",
) -> str:
    f = minimal_one_line(findings)
    i = minimal_one_line(impression)
    fmt = _normalize_target_format(target_format)

    if fmt == "findings":
        return _append_target_terminator(f, tokenizer)

    body = f"{f} impression: {i}".strip() if i else f"{f}".strip()
    return _append_target_terminator(body, tokenizer)


def build_full_target(
    findings: str,
    impression: str,
    tokenizer=None,
    target_format: str | None = "findings + impression",
) -> str:
    return build_target_text(
        findings=findings,
        impression=impression,
        tokenizer=tokenizer,
        target_format=target_format,
    )