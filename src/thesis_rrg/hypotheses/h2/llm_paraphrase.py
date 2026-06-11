from __future__ import annotations

import gc
import os
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import AutoConfig, BertModel, BertTokenizer
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

def _env_flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0") == "1"


SEED = int(os.environ.get("SEED", "1337"))

AUX_DEVICE = os.environ.get("AUX_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
if (not torch.cuda.is_available()) and AUX_DEVICE.startswith("cuda"):
    AUX_DEVICE = "cpu"

DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("DEFAULT_MAX_NEW_TOKENS", "220"))
DEFAULT_MIN_NEW_TOKENS = int(os.environ.get("DEFAULT_MIN_NEW_TOKENS", "16"))
DEFAULT_TEMPERATURE = float(os.environ.get("DEFAULT_TEMPERATURE", "0.8"))
DEFAULT_TOP_P = float(os.environ.get("DEFAULT_TOP_P", "0.95"))
DEFAULT_REP_PEN = float(os.environ.get("DEFAULT_REP_PEN", "1.10"))
DEFAULT_NO_REPEAT_NGRAM = int(os.environ.get("DEFAULT_NO_REPEAT_NGRAM", "4"))
DEFAULT_NUM_CANDS = int(os.environ.get("DEFAULT_NUM_CANDS", "8"))
DEFAULT_BATCH_SIZE = int(os.environ.get("DEFAULT_BATCH_SIZE", "1"))

USE_QUICK_SAFETY = _env_flag("USE_QUICK_SAFETY", True)
USE_LEXICAL_FILTER = _env_flag("USE_LEXICAL_FILTER", True)
USE_ASCII_FILTER = _env_flag("USE_ASCII_FILTER", True)
USE_FORMAT_FILTER = _env_flag("USE_FORMAT_FILTER", True)

# По умолчанию отключено (дорого), включается из prepare.py через nb.USE_RADGRAPH = True/False
USE_RADGRAPH = _env_flag("USE_RADGRAPH", False)
USE_CHEXBERT = _env_flag("USE_CHEXBERT", True)
USE_MEDSIGLIP = _env_flag("USE_MEDSIGLIP", True)

LEX_MAX_SEQ_RATIO = float(os.environ.get("LEX_MAX_SEQ_RATIO", "0.92"))
LEX_MAX_4GRAM_OVERLAP = float(os.environ.get("LEX_MAX_4GRAM_OVERLAP", "0.85"))
LEX_MIN_TOKEN_CHANGE = float(os.environ.get("LEX_MIN_TOKEN_CHANGE", "0.12"))

FORMAT_MIN_CHAR_RATIO = float(os.environ.get("FORMAT_MIN_CHAR_RATIO", "0.55"))
FORMAT_MAX_CHAR_RATIO = float(os.environ.get("FORMAT_MAX_CHAR_RATIO", "1.80"))

E2A_RG_MODE = os.environ.get("E2_RG_MODE", "threshold")
MIN_ENT_P = float(os.environ.get("E2_MIN_ENT_P", "0.95"))
MIN_ENT_R = float(os.environ.get("E2_MIN_ENT_R", "0.90"))
MIN_REL_P = float(os.environ.get("E2_MIN_REL_P", "0.85"))
MIN_REL_R = float(os.environ.get("E2_MIN_REL_R", "0.70"))

MAX_EXTRA_ENT = int(os.environ.get("E2_MAX_EXTRA_ENT", "2"))
MAX_MISSING_ENT = int(os.environ.get("E2_MAX_MISSING_ENT", "2"))
MAX_EXTRA_REL = int(os.environ.get("E2_MAX_EXTRA_REL", "4"))
MAX_MISSING_REL = int(os.environ.get("E2_MAX_MISSING_REL", "2"))

RADGRAPH_MODEL_TYPE = os.environ.get("RADGRAPH_MODEL_TYPE", "modern-radgraph-xl")

MEDSIGLIP_MODEL_ID = os.environ.get("MEDSIGLIP_MODEL_ID", "google/medsiglip-448")
MEDSIGLIP_MODE = os.environ.get("MEDSIGLIP_MODE", "impression")  # impression | sentence
MEDSIGLIP_MIN_SIM = float(os.environ.get("MEDSIGLIP_MIN_SIM", "0.75"))
MEDSIGLIP_MAX_SIM = float(os.environ.get("MEDSIGLIP_MAX_SIM", "0.985"))

CHEXBERT_USE_IMPRESSION_ONLY = _env_flag("CHEXBERT_USE_IMPRESSION_ONLY", True)
CHEXBERT_MAX_EXTRA_POS = int(os.environ.get("CHEXBERT_MAX_EXTRA_POS", "0"))
CHEXBERT_MAX_MISS_POS = int(os.environ.get("CHEXBERT_MAX_MISS_POS", "1"))
CHEXBERT_MAX_HAMMING = int(os.environ.get("CHEXBERT_MAX_HAMMING", "2"))
CHEXBERT_BATCH_SIZE = int(os.environ.get("CHEXBERT_BATCH_SIZE", "64"))
CHEXBERT_MAX_LEN = int(os.environ.get("CHEXBERT_MAX_LEN", "512"))
CHEXBERT_MODE = os.environ.get("CHEXBERT_MODE", "rrg")  # rrg | classification
CHEXBERT_DEVICE = AUX_DEVICE

LLM_DTYPE = os.environ.get("LLM_DTYPE", "bfloat16")
LLM_MAX_GPU_MEM_GIB = int(os.environ.get("LLM_MAX_GPU_MEM_GIB", "44"))
LLM_ATTN_IMPL = os.environ.get("LLM_ATTN_IMPL", "")
LLM_BATCH_SIZE = int(os.environ.get("LLM_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))

LLM_NUM_GPUS = int(os.environ.get("LLM_NUM_GPUS", "1"))
LLM_DEVICE = os.environ.get("LLM_DEVICE", "auto" if LLM_NUM_GPUS >= 2 else ("cuda:0" if torch.cuda.is_available() else "cpu"))
LLM_GPU_IDS = os.environ.get("LLM_GPU_IDS", "0" if torch.cuda.is_available() else "").replace(" ", "")
LLM_GPU_IDS_LIST = [int(x) for x in LLM_GPU_IDS.split(",") if x != ""]

if len(LLM_GPU_IDS_LIST) >= 2:
    LLM_DEVICE = "auto"
    LLM_NUM_GPUS = len(LLM_GPU_IDS_LIST)
elif len(LLM_GPU_IDS_LIST) == 1 and torch.cuda.is_available():
    LLM_NUM_GPUS = 1
    # Важно: при CUDA_VISIBLE_DEVICES remap это должен быть локальный индекс (обычно 0/1)
    LLM_DEVICE = os.environ.get("LLM_DEVICE", f"cuda:{LLM_GPU_IDS_LIST[0]}")

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


set_seed(SEED)

# Глобальные lazy-инстансы
rg = None
medsiglip = None
chexbert_runner = None
paraphraser = None

# ==============================================================================
# Text utils / prompt / parsing
# ==============================================================================

_ws = re.compile(r"\s+")
def minimal_one_line(text):
    if not isinstance(text, str):
        return ""
    return _ws.sub(" ", text.strip())

# Cell 3 — Utils: normalize / full / prompt / parsing

def minimal_one_line(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\\n", " ").replace("\\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_target_text(findings: str, impression: str) -> str:
    f = minimal_one_line(findings)
    i = minimal_one_line(impression)
    return f"{f} impression: {i}".strip() if i else f"{f}".strip()

LLM_SYSTEM = (
    "You are a radiology report rewriting assistant. "
    "Rewrite text while preserving ALL clinical facts exactly."
)

def build_paraphrase_prompt(findings: str, impression: str) -> str:
    f = minimal_one_line(findings)
    i = minimal_one_line(impression)
    return (
        "TASK:\n"
        "Rewrite the FINDINGS and IMPRESSION using different wording while preserving ALL facts exactly.\n\n"
        "HARD RULES:\n"
        "- Do NOT add facts.\n"
        "- Do NOT remove facts.\n"
        "- Do NOT change negations, uncertainty, laterality, locations, severity, or temporal change.\n"
        "- Do NOT add patient names, dates, IDs, headers, explanations, examples, or extra reports.\n"
        "- Use concise professional radiology-report language, not lay explanations.\n"
        "- Keep the length similar.\n\n"
        "OUTPUT FORMAT:\n"
        "Return exactly these two lines and stop:\n"
        "FINDINGS: <rewritten findings>\n"
        "IMPRESSION: <rewritten impression>\n\n"
        "INPUT:\n"
        f"FINDINGS: {f}\n"
        f"IMPRESSION: {i}\n"
    )

import re
from typing import Tuple

def _normalize_punct(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("：", ":").replace("﹕", ":").replace("；", ":")
    s = s.replace("。", ".").replace("，", ",")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\\+\s*\n", "\n", s)
    s = re.sub(r":{2,}", ":", s)
    return s

_SPECIAL_TOK_RE = re.compile(r"(?is)<\s*unused\d+\s*>|<\|.*?\|>|```.*?```")
_BULLET_PREFIX_RE = re.compile(r"(?m)^\s*([-*]|\d+\)|\d+\.)\s+")

# Важно: не ловим IMPROVEMENT, только варианты IMPR* хедеров
_IMP_HDR_RE = re.compile(
    r"(?i)\b(?:"
    r"IMPRESSION(?:\s*[_-]\s*(?:REVISED|REV|FINAL|OUTPUT|\d+))?"
    r"|IMPRESSION_REVISED"
    r"|IMPRESSION_"
    r"|IMPRESSIONS"
    r"|IMPRESS\s+ION"
    r"|IMPREESSION"
    r"|IMPESSION"
    r"|IMPRESION"
    r"|IMPRISSION"
    r"|IMPRESSSION"
    r"|IMPRSSION|IMPRSSON|IMPRSSIONS"
    r"|IMMPRESSION"
    r")\b"
)
_FIND_HDR_RE = re.compile(r"(?i)\bFINDINGS?\b")

def _norm_keep_newlines(s: str) -> str:
    s = _normalize_punct(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s).strip()
    return s

def strip_special_tokens(s: str) -> str:
    if not s:
        return ""
    s = _SPECIAL_TOK_RE.sub(" ", s)
    s = re.sub(r"(?is)<\s*unused\d+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_two_lines(text: str) -> Tuple[str, str]:
    t = _norm_keep_newlines(text)
    if not t:
        return "", ""

    t = re.sub(r"(?is)<think>.*?</think>", " ", t)

    m0 = _FIND_HDR_RE.search(t)
    if m0:
        t = t[m0.start():]

    t = re.sub(r"(?i)\bFINDINGS?\s*[\.;:\-]+", "FINDINGS:", t)
    t = _IMP_HDR_RE.sub("IMPRESSION", t)
    t = re.sub(r"(?i)\bIMPRESSION\s*[\.;:\-]+", "IMPRESSION:", t)

    m = re.search(r"(?is)\bFINDINGS\s*[:\-]\s*(.*?)\s*\bIMPRESSION\s*[:\-]\s*(.*)", t)
    if not m:
        return "", ""

    f = m.group(1).strip()
    rest = m.group(2).strip()

    imp = rest.split("\n", 1)[0].strip() if "\n" in rest else rest
    imp = re.split(r"(?i)\bFINDINGS?\b\s*[:\-]", imp)[0].strip()

    f = _BULLET_PREFIX_RE.sub("", f)
    imp = _BULLET_PREFIX_RE.sub("", imp)

    f = strip_special_tokens(f)
    imp = strip_special_tokens(imp)

    f = re.sub(r"\s+", " ", f).strip()
    imp = re.sub(r"\s+", " ", imp).strip()

    if len(f) < 3 or len(imp) < 3:
        return "", ""
    return f, imp

def _extract_set(text: str, patterns: Dict[str, str]) -> set:
    s = set()
    t = " " + str(text).lower() + " "
    for k, pat in patterns.items():
        if re.search(pat, t):
            s.add(k)
    return s

BULLET_PREFIX_RE = re.compile(r"(?m)^\s*([-*]|\d+\)|\d+\.)\s+")
MULTISPACE_RE = re.compile(r"\s+")

TOK_CLEAN_RE = re.compile(r"[^a-z0-9]+")

# ==============================================================================
# Filters: quick / ascii / lexical
# ==============================================================================


_LAT_PATTERNS = {
    "left":   r"\b(left|lt)\b",
    "right":  r"\b(right|rt)\b",
    "bilateral": r"\b(bilateral|bilaterally|both|bibasilar|bibasal|biapical|bihilar)\b",
}

_SEV_PATTERNS = {
    "mild": r"\bmild(?:ly)?\b",
    "moderate": r"\bmoderate(?:ly)?\b",
    "severe": r"\bsevere(?:ly)?\b",
    "minimal": r"\bminimal(?:ly)?\b",
    "marked": r"\bmarked(?:ly)?\b",
}

QUICK_STRICT_SEVERITY = _env_flag("QUICK_STRICT_SEVERITY", False)

def _laterality_hard_mismatch(lat_src: set, lat_cand: set) -> bool:
    """Catch obvious unilateral left/right flips without rejecting benign mentions.

    A whole-report laterality set is too crude for reports that say, for example,
    "right upper mass" and "left lung is clear"; dropping the normal-side mention
    should be a warning, not an automatic reject. We keep hard rejection for the
    most dangerous case: a single-sided source finding is missing or flipped.
    """
    src_sides = lat_src & {"left", "right"}
    cand_sides = lat_cand & {"left", "right"}
    if len(src_sides) != 1 or "bilateral" in lat_src:
        return False

    expected = next(iter(src_sides))
    opposite = "left" if expected == "right" else "right"
    return expected not in cand_sides or opposite in cand_sides

def _extract_set(text: str, patterns: Dict[str, str]) -> set:
    s = set()
    t = " " + str(text).lower() + " "
    for k, pat in patterns.items():
        if re.search(pat, t):
            s.add(k)
    return s


def quick_safety_ok(src_full: str, cand_full: str) -> Tuple[bool, Dict[str, Any]]:
    """Cheap guardrail.

    Hard fail:
      - laterality mismatch (left/right/bilateral)

    Severity:
      - by default it's only a WARNING (does not fail),
        because in your benchmark quick filter was the biggest bottleneck.
      - set QUICK_STRICT_SEVERITY=1 to make severity mismatch a hard fail.
    """
    lat_src = _extract_set(src_full, _LAT_PATTERNS)
    lat_cand = _extract_set(cand_full, _LAT_PATTERNS)
    lat_mismatch = lat_src != lat_cand
    if lat_mismatch and _laterality_hard_mismatch(lat_src, lat_cand):
        return False, {
            "lat_src": lat_src, "lat_cand": lat_cand,
            "sev_src": set(), "sev_cand": set(),
            "sev_mismatch": False,
            "reason": "laterality_mismatch"
        }

    sev_src = _extract_set(src_full, _SEV_PATTERNS)
    sev_cand = _extract_set(cand_full, _SEV_PATTERNS)
    sev_mismatch = (sev_src != sev_cand)

    info = {
        "lat_src": lat_src, "lat_cand": lat_cand,
        "sev_src": sev_src, "sev_cand": sev_cand,
        "sev_mismatch": sev_mismatch,
        "reason": (
            "severity_mismatch"
            if sev_mismatch
            else ("laterality_warning" if lat_mismatch else "ok")
        )
    }

    if sev_mismatch and QUICK_STRICT_SEVERITY:
        return False, info

    # severity mismatch is a warning by default
    return True, info

def ascii_only_ok(text: str) -> Tuple[bool, Dict[str, Any]]:
    bad = [ch for ch in text if ord(ch) >= 128]
    if bad:
        sample = "".join(sorted(set(bad)))[:10]
        return False, {"reason": "non_ascii", "bad_chars": sample}
    return True, {"reason": "ok"}


_word_re = re.compile(r"[A-Za-z0-9]+|[^\\sA-Za-z0-9]")

def _tok(s: str):
    return _word_re.findall((s or "").lower())

def _ngram_set(tokens, n=4):
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

def lexical_change_ok(src_full: str, cand_full: str,
                      max_seq_ratio: float = LEX_MAX_SEQ_RATIO,
                      max_4gram_overlap: float = LEX_MAX_4GRAM_OVERLAP,
                      min_token_change: float = LEX_MIN_TOKEN_CHANGE) -> Tuple[bool, Dict[str, Any]]:
    a = minimal_one_line(src_full)
    b = minimal_one_line(cand_full)
    seq_ratio = SequenceMatcher(None, a, b).ratio()
    ta, tb = _tok(a), _tok(b)
    sa, sb = set(ta), set(tb)
    na, nb = _ngram_set(ta, 4), _ngram_set(tb, 4)
    overlap4 = (len(na & nb) / max(1, len(na | nb))) if (na or nb) else 0.0
    changed_frac = 1.0 - (len(sa & sb) / max(1, len(sa)))
    info = dict(seq_ratio=float(seq_ratio), overlap4=float(overlap4), changed_frac=float(changed_frac))
    if seq_ratio > max_seq_ratio:
        return False, {**info, "reason":"seq_ratio_high"}
    if overlap4 > max_4gram_overlap:
        return False, {**info, "reason":"4gram_overlap_high"}
    if changed_frac < min_token_change:
        return False, {**info, "reason":"too_few_token_changes"}
    return True, {**info, "reason":"ok"}


# ==============================================================================
# RadGraph
# ==============================================================================

BULLET_PREFIX_RE = re.compile(r"(?m)^\s*([-*]|\d+\)|\d+\.)\s+")
MULTISPACE_RE = re.compile(r"\s+")

def canonical_for_radgraph(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    t = BULLET_PREFIX_RE.sub("", t)
    t = t.replace("\n", ". ")
    t = re.sub(r"\s*;\s*", ". ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

TOK_CLEAN_RE = re.compile(r"[^a-z0-9]+")

def clean_tok(x: str) -> str:
    return TOK_CLEAN_RE.sub("", str(x).lower().strip())

def norm_tokens(tok):
    if tok is None:
        return tuple()
    if isinstance(tok, list):
        out = []
        for t in tok:
            ct = clean_tok(t)
            if ct:
                out.append(ct)
        return tuple(out)
    if isinstance(tok, str):
        parts = [clean_tok(p) for p in tok.split(" ")]
        parts = [p for p in parts if p]
        return tuple(parts)
    ct = clean_tok(tok)
    return tuple([ct]) if ct else tuple()

def rg_to_sets_v2(ann_dict):
    if not isinstance(ann_dict, dict) or len(ann_dict) == 0:
        return set(), set()

    k0 = next(iter(ann_dict.keys()))
    payload = ann_dict[k0]

    for _ in range(4):
        if isinstance(payload, dict) and "entities" in payload:
            break
        if isinstance(payload, dict) and len(payload) == 1:
            payload = next(iter(payload.values()))
            continue
        if isinstance(payload, dict) and "0" in payload and isinstance(payload["0"], dict):
            payload = payload["0"]
            continue
        break

    if not (isinstance(payload, dict) and "entities" in payload):
        return set(), set()

    entities = payload.get("entities", {}) or {}

    ent_map = {}
    ent_set = set()
    for ent_id, ent in entities.items():
        lab = str(ent.get("label", "")).strip()
        tok = norm_tokens(ent.get("tokens", []))
        key = (tok, lab)
        ent_map[str(ent_id)] = key
        ent_set.add(key)

    rel_set = set()
    for src_id, ent in entities.items():
        src_key = ent_map.get(str(src_id), None)
        if src_key is None:
            continue
        rels = ent.get("relations", []) or []
        for rel in rels:
            if not isinstance(rel, (list, tuple)) or len(rel) != 2:
                continue
            rel_type, tgt_id = str(rel[0]), str(rel[1])
            tgt_key = ent_map.get(str(tgt_id), None)
            if tgt_key is None:
                continue
            rel_set.add((src_key, rel_type, tgt_key))

    return ent_set, rel_set

def pr(ref_set, hyp_set):
    if len(ref_set) == 0 and len(hyp_set) == 0:
        return 1.0, 1.0
    if len(hyp_set) == 0:
        return 0.0, 0.0
    inter = len(ref_set & hyp_set)
    p = inter / max(1, len(hyp_set))
    r = inter / max(1, len(ref_set))
    return float(p), float(r)

def radgraph_pair_stats(src: str, cand: str):
    assert rg is not None
    src_c = canonical_for_radgraph(src)
    cand_c = canonical_for_radgraph(cand)

    hyp_anns = rg([cand_c])
    ref_anns = rg([src_c])

    hyp_ent, hyp_rel = rg_to_sets_v2(hyp_anns)
    ref_ent, ref_rel = rg_to_sets_v2(ref_anns)

    ent_p, ent_r = pr(ref_ent, hyp_ent)
    rel_p, rel_r = pr(ref_rel, hyp_rel)

    return {
        "ent_p": ent_p, "ent_r": ent_r,
        "rel_p": rel_p, "rel_r": rel_r,
        "ref_ent": ref_ent, "hyp_ent": hyp_ent,
        "ref_rel": ref_rel, "hyp_rel": hyp_rel,
    }

def radgraph_ok(st):
    extra_ent = len(st["hyp_ent"] - st["ref_ent"])
    miss_ent  = len(st["ref_ent"] - st["hyp_ent"])
    extra_rel = len(st["hyp_rel"] - st["ref_rel"])
    miss_rel  = len(st["ref_rel"] - st["hyp_rel"])

    if E2A_RG_MODE == "strict":
        if extra_ent != 0:
            return False, "extra_entities"
        if miss_ent != 0:
            return False, "missing_entities"
        if miss_rel != 0:
            return False, "missing_relations"
        if extra_rel > MAX_EXTRA_REL:
            return False, "too_many_extra_relations"
        if st["rel_p"] < MIN_REL_P:
            return False, "rel_precision_low"
        if st["ent_p"] < MIN_ENT_P or st["ent_r"] < MIN_ENT_R:
            return False, "entity_pr_low"
        return True, "ok"

    # if st["ent_p"] < MIN_ENT_P:
    #     return False, "ent_precision_low"
    if st["ent_r"] < MIN_ENT_R:
        return False, "ent_recall_low"
    if st["rel_p"] < MIN_REL_P:
        return False, "rel_precision_low"
    if st["rel_r"] < MIN_REL_R:
        return False, "rel_recall_low"

    if extra_ent > MAX_EXTRA_ENT:
        return False, "too_many_extra_entities"
    if miss_ent > MAX_MISSING_ENT:
        return False, "too_many_missing_entities"
    if miss_rel > MAX_MISSING_REL:
        return False, "too_many_missing_relations"
    if extra_rel > MAX_EXTRA_REL:
        return False, "too_many_extra_relations"

    return True, "ok"
    
rg = None

def try_init_radgraph():
    global rg
    if not USE_RADGRAPH:
        print("[RadGraph] disabled by config")
        return None
    try:
        from radgraph import RadGraph
        # RadGraph takes cuda index (int) in many versions; we support 'cuda:X' too
        cuda_idx = None
        if AUX_DEVICE.startswith("cuda:"):
            cuda_idx = int(AUX_DEVICE.split(":")[1])
        elif AUX_DEVICE == "cpu":
            cuda_idx = -1
        else:
            cuda_idx = 0
        print("[RadGraph] init model_type:", RADGRAPH_MODEL_TYPE, "cuda:", cuda_idx)
        rg = RadGraph(model_type=RADGRAPH_MODEL_TYPE, cuda=cuda_idx)
        print("[RadGraph] loaded OK")
        return rg
    except Exception as e:
        print("[RadGraph] init failed:", repr(e))
        rg = None
        return None


# ==============================================================================
# CheXbert
# ==============================================================================

CHEXBERT_CONDITIONS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CHEXBERT_NO_FINDING = "No Finding"
CHEXBERT_TARGETS = CHEXBERT_CONDITIONS + [CHEXBERT_NO_FINDING]
_CHEXBERT_CACHE_DIR = os.environ.get(
    "CHEXBERT_CACHE_DIR",
    str(Path.home() / ".cache" / "chexbert"),
)
Path(_CHEXBERT_CACHE_DIR).mkdir(parents=True, exist_ok=True)
print("[CheXbert] cache_dir:", _CHEXBERT_CACHE_DIR)

def _attention_mask_from_input_ids(input_ids: torch.LongTensor) -> torch.FloatTensor:
    # BertTokenizer pad_token_id == 0 for bert-base-uncased
    return (input_ids != 0).float()


class CheXbertBertLabeler(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.device = torch.device(device)

        cfg = AutoConfig.from_pretrained("bert-base-uncased")
        self.bert = BertModel(cfg)
        hidden = self.bert.config.hidden_size

        # было: self.heads = nn.ModuleList(...)
        self.linear_heads = nn.ModuleList([nn.Linear(hidden, 4) for _ in range(13)])
        self.linear_heads.append(nn.Linear(hidden, 2))

        self.dropout = nn.Dropout(0.1)

        ckpt_path = hf_hub_download(
            repo_id="StanfordAIMI/RRG_scorers",
            filename="chexbert.pth",
            cache_dir=_CHEXBERT_CACHE_DIR,
        )
        state = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}

        self.load_state_dict(state, strict=True)

        self.to(self.device)
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward_logits(self, input_ids: torch.LongTensor) -> List[torch.Tensor]:
        attn = _attention_mask_from_input_ids(input_ids)
        out = self.bert(input_ids=input_ids, attention_mask=attn)
        cls = self.dropout(out.last_hidden_state[:, 0])  # [B, H]

        # было: for h in self.heads
        return [h(cls) for h in self.linear_heads]  # list of [B, num_classes]


class CheXbertRunner:
    def __init__(self, device: str = CHEXBERT_DEVICE):
        self.device = torch.device(device)
        self.tok = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = CheXbertBertLabeler(device=str(self.device))

    @torch.no_grad()
    def label_many(self, texts: Sequence[str], mode: str = CHEXBERT_MODE, batch_size: int = CHEXBERT_BATCH_SIZE) -> np.ndarray:
        """
        Returns labels:
          - mode="rrg": бинарные {0,1}, shape [N,14]
          - mode="classification": {-1,0,1}, shape [N,14]
        """
        texts = ["" if t is None else str(t) for t in texts]
        all_out = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tok(
                batch,
                padding=True,
                truncation=True,
                max_length=CHEXBERT_MAX_LEN,
                return_tensors="pt",
            )
            input_ids = enc.input_ids.to(self.device)

            logits_list = self.model.forward_logits(input_ids)
            # argmax per head: list of [B]
            preds = [lg.argmax(dim=1).detach().cpu().numpy() for lg in logits_list]  # len=14
            preds = np.stack(preds, axis=1)  # [B,14]

            if mode == "rrg":
                # in F1CheXbert mapping: treat classes {1,3} as positive (positive OR uncertain) :contentReference[oaicite:2]{index=2}
                bin_out = np.isin(preds, [1, 3]).astype(np.int64)
                all_out.append(bin_out)
            elif mode == "classification":
                # map to {-1,0,1}. This mapping mirrors the common CheXbert conventions:
                # class 1 => positive, class 2 => negative, class 3 => uncertain, else => blank(0)
                out = np.zeros_like(preds, dtype=np.int64)
                out[preds == 1] = 1
                out[preds == 3] = -1
                out[preds == 2] = 0
                all_out.append(out)
            else:
                raise ValueError(f"Unknown CheXbert mode: {mode}")

        return np.concatenate(all_out, axis=0) if all_out else np.zeros((0, 14), dtype=np.int64)


def _extract_impression_from_full(full_text: str) -> str:
    if not full_text:
        return ""
    t = str(full_text).replace("\r", "\n")
    m = re.search(r"(?is)\bimpression\s*[:\-]\s*(.*)$", t)
    if m:
        imp = m.group(1).strip()
        # обрежем если вдруг после <unused..> начался новый FINDINGS
        imp = re.split(r"(?is)\bfindings\s*[:\-]", imp)[0].strip()
        return imp
    # fallback: если не нашли разделитель — считаем весь текст impression-ом
    return t.strip()


def chexbert_ok(src_full: str, cand_full: str) -> Tuple[bool, Dict[str, Any]]:
    if chexbert_runner is None:
        return False, {"reason": "chexbert_not_loaded"}

    if CHEXBERT_USE_IMPRESSION_ONLY:
        src_txt = "IMPRESSION: " + _extract_impression_from_full(src_full)
        cand_txt = "IMPRESSION: " + _extract_impression_from_full(cand_full)
    else:
        # лучше форматировать с явными заголовками
        src_txt = str(src_full)
        cand_txt = str(cand_full)

    lbl = chexbert_runner.label_many([src_txt, cand_txt], mode="rrg", batch_size=2)  # [2,14]
    if lbl.shape[0] != 2:
        return False, {"reason": "chexbert_bad_shape", "shape": tuple(lbl.shape)}

    src = lbl[0].astype(np.int64)
    cand = lbl[1].astype(np.int64)

    extra_pos = int(((cand == 1) & (src == 0)).sum())
    miss_pos  = int(((cand == 0) & (src == 1)).sum())
    hamming   = int((cand != src).sum())

    ok = (
        extra_pos <= CHEXBERT_MAX_EXTRA_POS
        and miss_pos <= CHEXBERT_MAX_MISS_POS
        and hamming <= CHEXBERT_MAX_HAMMING
    )

    info = {
        "extra_pos": extra_pos,
        "miss_pos": miss_pos,
        "hamming": hamming,
        "src_pos_cnt": int((src == 1).sum()),
        "cand_pos_cnt": int((cand == 1).sum()),
        "reason": "ok" if ok else "label_mismatch",
    }
    return ok, info


# ==============================================================================
# MedSigLIP
# ==============================================================================


def extract_impression_only(full: str) -> str:
    t = minimal_one_line(full)
    m = re.search(r"(?i)\\bimpression\\s*:\\s*(.*)$", t)
    return minimal_one_line(m.group(1)) if m else t

_SENT_SPLIT = re.compile(r"(?<=[\\.!?;])\\s+")

def extract_findings_only(full: str) -> str:
    t = minimal_one_line(full)
    m = re.search(r"(?is)^(.*?)\\bimpression\\s*:\\s*", t)
    return minimal_one_line(m.group(1)) if m else t

def try_init_medsiglip():
    global medsiglip
    if not USE_MEDSIGLIP:
        print("[MedSigLIP] disabled by config")
        return None
    try:
        from transformers import AutoProcessor, SiglipModel
        print("[MedSigLIP] loading:", MEDSIGLIP_MODEL_ID, "device:", AUX_DEVICE)
        proc = AutoProcessor.from_pretrained(MEDSIGLIP_MODEL_ID)
        model = SiglipModel.from_pretrained(MEDSIGLIP_MODEL_ID).eval().to(AUX_DEVICE)
        medsiglip = (proc, model)
        return medsiglip
    except Exception as e:
        print("[MedSigLIP] init failed:", repr(e))
        medsiglip = None
        return None

@torch.inference_mode()
def _msl_embed_text(texts: List[str], max_length: int = 64) -> torch.Tensor:
    proc, model = medsiglip
    inputs = proc(text=[minimal_one_line(t) for t in texts],
                  return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(AUX_DEVICE)
    feats = model.get_text_features(**inputs)
    feats = F.normalize(feats, dim=-1)
    return feats

def medsiglip_ok(src_full: str, cand_full: str) -> Tuple[bool, Dict[str, Any]]:
    if medsiglip is None:
        return True, {"sim": None, "reason": "skipped"}
    if MEDSIGLIP_MODE == "sentence":
        a_s = [s.strip() for s in _SENT_SPLIT.split(extract_findings_only(src_full)) if s.strip()]
        b_s = [s.strip() for s in _SENT_SPLIT.split(extract_findings_only(cand_full)) if s.strip()]
        a_s.append(extract_impression_only(src_full))
        b_s.append(extract_impression_only(cand_full))
        feats = _msl_embed_text(a_s + b_s, max_length=64)
        a = feats[:len(a_s)].mean(dim=0, keepdim=True)
        b = feats[len(a_s):].mean(dim=0, keepdim=True)
        a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
        sim = float((a*b).sum().item())
    else:
        feats = _msl_embed_text([extract_impression_only(src_full), extract_impression_only(cand_full)], max_length=64)
        sim = float((feats[0]*feats[1]).sum().item())
    ok = (MEDSIGLIP_MIN_SIM <= sim <= MEDSIGLIP_MAX_SIM)
    return ok, {"sim": sim, "reason": "ok" if ok else "sim_out_of_range"}


# ==============================================================================
# Sharding helpers
# ==============================================================================
def add_shard_id(df_in: pd.DataFrame, num_shards: int = 4, key: str = "study_id") -> pd.DataFrame:
    df_out = df_in.copy()
    h = pd.util.hash_pandas_object(df_out[key], index=False).astype("uint64")
    df_out["shard_id"] = (h % np.uint64(num_shards)).astype(int)
    return df_out

def save_all_shards(df_in: pd.DataFrame, out_dir: Path, num_shards: int) -> List[Path]:
    out_paths = []
    for sid in range(num_shards):
        part = df_in[df_in["shard_id"] == sid].reset_index(drop=True)
        p = out_dir / f"mimic_clean_shard{sid}_of{num_shards}.parquet"
        part.to_parquet(p, index=False, engine="pyarrow", compression="snappy")
        print("Saved shard", sid, "rows:", len(part), "->", p)
        out_paths.append(p)
    return out_paths


# ==============================================================================
# LLM paraphraser
# ==============================================================================

@dataclass
class ModelSpec:
    model_id: str = "google/medgemma-4b-it"
    kind: str = "causal"  # causal|seq2seq
    use_chat_template: bool = True
    load_in_4bit: bool = False
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    min_new_tokens: int = DEFAULT_MIN_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repetition_penalty: float = DEFAULT_REP_PEN
    no_repeat_ngram: int = DEFAULT_NO_REPEAT_NGRAM
    n_candidates: int = DEFAULT_NUM_CANDS
    input_max_length: int = 1024
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class HFParaphraser:
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.model_name = spec.model_id
        self.model_type = spec.kind

        print("[LLM] Loading tokenizer:", self.model_name)
        self.tok = AutoTokenizer.from_pretrained(
            self.model_name, 
            use_fast=True, 
            trust_remote_code=spec.trust_remote_code
        )

        if getattr(self.tok, "pad_token_id", None) is None:
            self.tok.pad_token = (
                self.tok.eos_token if getattr(self.tok, "eos_token", None) is not None else self.tok.unk_token
            )

        if self.model_type == "causal":
            try:
                self.tok.padding_side = "left"
            except Exception:
                pass

        def _dtype_from_str(s: str):
            s = (s or "").lower().strip()
            if s in {"bf16", "bfloat16"}:
                return torch.bfloat16
            if s in {"fp16", "float16", "half"}:
                return torch.float16
            if s in {"fp32", "float32"}:
                return torch.float32
            return "auto"

        torch_dtype = _dtype_from_str(spec.dtype)

        if "LLM_GPU_IDS_LIST" in globals() and len(LLM_GPU_IDS_LIST) > 0:
            target_gpus = list(LLM_GPU_IDS_LIST)
        else:
            target_gpus = list(range(torch.cuda.device_count()))
        # GPU Handling logic from original snippet
        gpu_ids = list(LLM_GPU_IDS_LIST) if ("LLM_GPU_IDS_LIST" in globals() and len(LLM_GPU_IDS_LIST) > 0) else list(range(torch.cuda.device_count()))
        if 'LLM_NUM_GPUS' in globals() and LLM_NUM_GPUS and LLM_NUM_GPUS > 0:
            gpu_ids = gpu_ids[:LLM_NUM_GPUS]
        print(gpu_ids)
        # Device Map Logic
        use_device_map = ('LLM_DEVICE' in globals() and LLM_DEVICE == "auto") or (len(gpu_ids) >= 2)
        device_map = "auto" if use_device_map else None

        max_memory = None
        if use_device_map and torch.cuda.is_available():
            per_gpu = globals().get("LLM_MAX_GPU_MEM_GIB", 44)
            max_memory = {i: "1GiB" for i in range(torch.cuda.device_count())}
            for gid in target_gpus:
                max_memory[gid] = f"{per_gpu}GiB"
            max_memory["cpu"] = "64GiB"

        quant_cfg = None
        if spec.load_in_4bit:
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=(torch_dtype if torch_dtype != "auto" else torch.bfloat16),
            )

        print("[LLM] Loading model:", self.model_name, "type:", self.model_type, "device_map:", device_map, "max_memory:", max_memory)

        common_kwargs = dict(low_cpu_mem_usage=True, trust_remote_code=spec.trust_remote_code)
        if 'LLM_ATTN_IMPL' in globals() and LLM_ATTN_IMPL:
            common_kwargs["attn_implementation"] = LLM_ATTN_IMPL

        # Load Model
        if self.model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                max_memory=max_memory,
                torch_dtype=None if quant_cfg is not None else torch_dtype,
                quantization_config=quant_cfg,
                **common_kwargs,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                max_memory=max_memory,
                torch_dtype=None if quant_cfg is not None else torch_dtype,
                quantization_config=quant_cfg,
                **common_kwargs,
            )

        
        # Move to device if not using device_map
        print(use_device_map)
        print(spec.device)
        if not use_device_map:
            dev = torch.device(spec.device) if (spec.device and spec.device != "auto") else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(dev)
            self.model = self.model.to(dev)
            self._device = dev
        else:
            self._device = self._infer_input_device()

        self.model.eval()
        self.input_device = self._infer_input_device()
        self._eos_token_ids = self._collect_eos_token_ids()
        print("[LLM] Model loaded. input_device:", self.input_device)
        print("[LLM] EOS token ids:", self._eos_token_ids)

    def _infer_input_device(self) -> torch.device:
        hfdm = getattr(self.model, "hf_device_map", None)
        if isinstance(hfdm, dict) and len(hfdm) > 0:
            preferred_keys = [
                "model.embed_tokens",
                "language_model.model.embed_tokens",
                "transformer.wte",
                "embeddings.word_embeddings",
            ]
            for k in preferred_keys:
                if k in hfdm:
                    return torch.device(hfdm[k])
            return torch.device(list(hfdm.values())[0])
        return next(self.model.parameters()).device

    def _collect_eos_token_ids(self) -> List[int]:
        ids: List[int] = []
        base = getattr(self.tok, "eos_token_id", None)
        if isinstance(base, int):
            ids.append(base)
        elif isinstance(base, (list, tuple)):
            ids.extend(int(x) for x in base if x is not None)

        for tok in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>"):
            try:
                tid = self.tok.convert_tokens_to_ids(tok)
            except Exception:
                tid = None
            if isinstance(tid, int) and tid >= 0 and tid != getattr(self.tok, "unk_token_id", None):
                ids.append(tid)

        out = []
        seen = set()
        for tid in ids:
            if tid not in seen:
                out.append(tid)
                seen.add(tid)
        return out

    @torch.inference_mode()
    def generate(self, prompts: List[str], spec_override: Optional[ModelSpec] = None) -> List[List[str]]:
        """
        Returns list[list[str]]: N completions per prompt.
        Allows overriding generation specs per call.
        """
        current_spec = spec_override if spec_override else self.spec
        n_per_prompt = current_spec.n_candidates
        
        # Batching logic (using global batch size if available, else 1)
        batch_size = globals().get('LLM_BATCH_SIZE', 1)
        out = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]

            # Build inputs
            if current_spec.use_chat_template and hasattr(self.tok, "apply_chat_template"):
                conversations = []
                system_prompt = globals().get('LLM_SYSTEM', "You are a helpful assistant.")
                for p in batch_prompts:
                    conversations.append([
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": p},
                    ])

                inputs = self.tok.apply_chat_template(
                    conversations,
                    add_generation_prompt=True,
                    tokenize=True,
                    padding=True,
                    truncation=True,
                    max_length=current_spec.input_max_length,
                    return_dict=True,
                    return_tensors="pt",
                )
            else:
                # Fallback for base models without chat template
                system_prompt = globals().get('LLM_SYSTEM', "")
                texts = [f"{system_prompt}\n\n{p}" for p in batch_prompts]
                inputs = self.tok(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=current_spec.input_max_length,
                )

            input_len = inputs["input_ids"].shape[-1]
            inputs = {k: v.to(self.input_device) for k, v in inputs.items()}

            gen_kwargs = dict(
                max_new_tokens=current_spec.max_new_tokens,
                min_new_tokens=current_spec.min_new_tokens,
                do_sample=True,
                temperature=current_spec.temperature,
                top_p=current_spec.top_p,
                repetition_penalty=current_spec.repetition_penalty,
                num_return_sequences=n_per_prompt,
                pad_token_id=self.tok.pad_token_id,
            )
            if self._eos_token_ids:
                gen_kwargs["eos_token_id"] = self._eos_token_ids

            if current_spec.no_repeat_ngram > 0:
                gen_kwargs["no_repeat_ngram_size"] = current_spec.no_repeat_ngram

            if self.model_type == "seq2seq":
                gen_kwargs.pop("repetition_penalty", None)
                gen_kwargs.pop("no_repeat_ngram_size", None)

            gen_ids = self.model.generate(**inputs, **gen_kwargs)

            if self.model_type == "seq2seq":
                decoded = self.tok.batch_decode(gen_ids, skip_special_tokens=True)
            else:
                # causal: cut off prompt
                comp_ids = gen_ids[:, input_len:]
                decoded = self.tok.batch_decode(comp_ids, skip_special_tokens=True)

            # Regroup results per prompt
            for j in range(len(batch_prompts)):
                chunk = decoded[j * n_per_prompt:(j + 1) * n_per_prompt]
                # Keep raw newlines: parse_two_lines uses the first line after
                # IMPRESSION as a hard boundary against extra LLM continuations.
                out.append([x.strip() for x in chunk])

        return out

    def unload(self):
        del self.tok
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==============================================================================
# Candidate eval / selection
# ==============================================================================

_BAD_FORMAT_RE = re.compile(
    r"(?is)"
    r"```"
    r"|<\s*unused\d+"
    r"|<\|.*?\|>"
    r"|<start_of_image>"
    r"|\b(?:task|hard rules|input|output|rewrite|rewritten|paraphrase|rationale|explanation|hint)\s*:"
    r"|\b(?:case file|blockchain_tag|blocktext|tool_text|macro findings|endmacro)\b"
    r"|\b(?:patient|name|date of examination|study date|exam type|procedure|comparison)\s*:"
    r"|\b(?:findings?)\s*:"
)

_UNSUPPORTED_IF_NEW_RE = re.compile(r"(?i)\b(?:hemothorax|hydropneumothorax)\b")

def format_safety_ok(src_full: str, cand_full: str) -> Tuple[bool, Dict[str, Any]]:
    """Reject obvious parser/LLM artifacts before a candidate can be selected."""
    cand = "" if cand_full is None else str(cand_full).strip()
    src = "" if src_full is None else str(src_full).strip()

    if not cand:
        return False, {"reason": "empty_candidate"}

    impression_markers = len(re.findall(r"(?i)\bimpression\s*:", cand))
    if impression_markers != 1:
        return False, {"reason": "bad_impression_marker_count", "impression_markers": impression_markers}

    if _BAD_FORMAT_RE.search(cand):
        return False, {"reason": "artifact_or_nested_header", "impression_markers": impression_markers}

    unsupported_new = sorted(
        {
            m.group(0).lower()
            for m in _UNSUPPORTED_IF_NEW_RE.finditer(cand)
            if not re.search(rf"(?i)\b{re.escape(m.group(0))}\b", src)
        }
    )
    if unsupported_new:
        return False, {
            "reason": "unsupported_new_term",
            "unsupported_new_terms": ",".join(unsupported_new),
            "impression_markers": impression_markers,
        }
        
    src_len = max(1, len(minimal_one_line(src)))
    cand_len = len(minimal_one_line(cand))
    ratio = cand_len / src_len
    if ratio < FORMAT_MIN_CHAR_RATIO:
        return False, {"reason": "too_short", "char_ratio": float(ratio), "impression_markers": impression_markers}
    if ratio > FORMAT_MAX_CHAR_RATIO:
        return False, {"reason": "too_long", "char_ratio": float(ratio), "impression_markers": impression_markers}

    return True, {"reason": "ok", "char_ratio": float(ratio), "impression_markers": impression_markers}


def eval_candidate_filters(src_full: str, cand_full: str) -> Dict[str, Any]:
    # Keep defaults explicit so a disabled/reordered gate cannot turn into a
    # KeyError and poison an entire shard with `filter_failed`.
    res: Dict[str, Any] = {
        "quick_ok": True,
        "quick_reason": "not_evaluated",
        "lex_ok": True,
        "lex_reason": "not_evaluated",
        "ascii_ok": True,
        "ascii_reason": "not_evaluated",
        "ascii_bad_chars": "",
        "format_ok": True,
        "format_reason": "not_evaluated",
        "chex_ok": True,
        "chex_reason": "not_evaluated",
        "rg_ok": True,
        "rg_reason": "not_evaluated",
        "msl_ok": True,
        "msl_sim": np.nan,
        "msl_reason": "not_evaluated",
    }

    if USE_QUICK_SAFETY:
        ok, info = quick_safety_ok(src_full, cand_full)
        res.update({"quick_ok": ok, **{f"quick_{k}": v for k, v in info.items()}})
    else:
        res["quick_ok"] = True

    if USE_LEXICAL_FILTER:
        ok, info = lexical_change_ok(src_full, cand_full)
        res.update({"lex_ok": ok, **{f"lex_{k}": v for k, v in info.items()}})
    else:
        res["lex_ok"] = True

    if USE_ASCII_FILTER:
        ok, info = ascii_only_ok(cand_full)
        res.update({"ascii_ok": ok, "ascii_reason": info.get("reason", ""), "ascii_bad_chars": info.get("bad_chars", "")})
    else:
        res["ascii_ok"] = True

    cheap_ok = bool(
        res.get("quick_ok", False)
        and res.get("lex_ok", False)
        and res.get("ascii_ok", False)
        and res.get("format_ok", False)
    )
    if not cheap_ok:
        if USE_CHEXBERT:
            res.update({"chex_ok": False, "chex_reason": "skipped_after_cheap_fail"})
        else:
            res.update({"chex_ok": True, "chex_reason": "skipped_by_config"})
        if USE_RADGRAPH:
            res.update({"rg_ok": False, "rg_reason": "skipped_after_cheap_fail"})
        else:
            res.update({"rg_ok": True, "rg_reason": "skipped_by_config"})
        if USE_MEDSIGLIP:
            res.update({"msl_ok": False, "msl_sim": np.nan, "msl_reason": "skipped_after_cheap_fail"})
        else:
            res.update({"msl_ok": True, "msl_sim": np.nan, "msl_reason": "skipped_by_config"})
        res["keep_ok"] = False
        res["all_filters_pass"] = False
        return res

    if USE_CHEXBERT:
        ok, info = chexbert_ok(src_full, cand_full)
        res.update({"chex_ok": ok, **{f"chex_{k}": v for k, v in info.items()}})
    else:
        res.update({"chex_ok": True, "chex_reason": "skipped_by_config"})

    if USE_RADGRAPH:
        if rg is None:
            res.update({"rg_ok": False, "rg_reason": "radgraph_not_loaded"})
        else:
            st = radgraph_pair_stats(src_full, cand_full)
            ok, reason = radgraph_ok(st)
            res.update({
                "rg_ok": ok,
                "rg_reason": reason,
                "rg_ent_p": st["ent_p"],
                "rg_ent_r": st["ent_r"],
                "rg_rel_p": st["rel_p"],
                "rg_rel_r": st["rel_r"],
            })
    else:
        res.update({"rg_ok": True, "rg_reason": "skipped_by_config"})

    ok, info = medsiglip_ok(src_full, cand_full)
    res.update({"msl_ok": ok, "msl_sim": info.get("sim", np.nan), "msl_reason": info.get("reason", "")})


    res["keep_ok"] = bool(
        res.get("quick_ok", False)
        and res.get("lex_ok", False)
        and res.get("chex_ok", False)
        and res.get("rg_ok", False)
        and res.get("msl_ok", False)
        and res.get("ascii_ok", False)
        and res.get("format_ok", False)
    )

    res["all_filters_pass"] = bool(res["keep_ok"])
    return res

# def run_paraphrase_with_filters(df_in: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     rows = []
#     t0 = time.time()

#     for _, r in tqdm(df_in.iterrows(), total=len(df_in), desc="E2c paraphrase"):
#         ex_id = f"{int(r['study_id'])}"
#         f0 = str(r["ref_findings"])
#         i0 = str(r["ref_impression"])
#         src_full = build_target_text(f0, i0)
#         prompt = build_paraphrase_prompt(f0, i0)

#         t_gen0 = time.time()
#         outs = paraphraser.generate([prompt])[0]
#         t_gen = time.time() - t_gen0

#         for k, raw in enumerate(outs):
#             pf, pi = parse_two_lines(raw)
#             parsed_ok = bool(pf and pi)
#             cand_full = build_target_text(pf, pi) if parsed_ok else ""

#             base = {
#                 "model_id": paraphraser.model_name,
#                 "ex_id": ex_id,
#                 "cand_id": k,
#                 "raw": raw,
#                 "parsed_ok": parsed_ok,
#                 "aug_findings": pf,
#                 "aug_impression": pi,
#                 "cand_full": cand_full,
#                 "src_full": src_full,
#                 "t_gen_s": float(t_gen),
#                 "subject_id": int(r["subject_id"]),
#                 "study_id": int(r["study_id"]),
#                 "dicom_id": str(r["dicom_id"]),
#                 "dcm_path": str(r["dcm_path"]),
#                 "split": str(r["split"]),
#                 "shard_id": int(r["shard_id"]),
#             }

#             if parsed_ok:
#                 filt = eval_candidate_filters(src_full, cand_full)
#             else:
#                 filt = {
#                     "quick_ok": False, "lex_ok": False, "chex_ok": False, "msl_ok": False,
#                     "msl_sim": np.nan, "ascii_ok": False,
#                     "keep_ok": False, "all_filters_pass": False
#                 }

#             rows.append({**base, **filt})

#     df_cands = pd.DataFrame(rows)

#     df_pass = df_cands[df_cands["keep_ok"] == True].copy()
#     if len(df_pass) == 0:
#         df_best = df_pass
#     else:
#         df_pass = df_pass.sort_values(["ex_id", "msl_sim"], ascending=[True, False])
#         df_best = df_pass.groupby("ex_id", as_index=False).head(1).reset_index(drop=True)

#     print("Candidates:", len(df_cands), "Passing (keep_ok):", len(df_pass), "Examples:", df_in["study_id"].nunique())
#     print("Time total (s):", time.time() - t0)
#     return df_cands, df_best


def apply_selection_gates(df_cands: pd.DataFrame) -> pd.DataFrame:
    """Re-apply cheap selection gates before choosing best candidates.

    This protects resume/reselect workflows from stale checkpoint chunks whose
    `keep_ok` column was computed with older filtering logic.
    """
    if len(df_cands) == 0:
        return df_cands

    df_out = df_cands.copy()
    index = df_out.index

    if {"src_full", "cand_full"}.issubset(df_out.columns):
        if USE_QUICK_SAFETY:
            quick_rows = []
            for src, cand in zip(df_out["src_full"], df_out["cand_full"]):
                ok, info = quick_safety_ok(src, cand)
                quick_rows.append({**info, "ok": ok})
            df_out["quick_ok"] = [bool(x["ok"]) for x in quick_rows]
            df_out["quick_reason"] = [x.get("reason", "") for x in quick_rows]
            df_out["quick_sev_mismatch"] = [bool(x.get("sev_mismatch", False)) for x in quick_rows]
        else:
            df_out["quick_ok"] = True

        if USE_FORMAT_FILTER:
            format_rows = [format_safety_ok(src, cand) for src, cand in zip(df_out["src_full"], df_out["cand_full"])]
            df_out["format_ok"] = [bool(ok) for ok, _ in format_rows]
            df_out["format_reason"] = [info.get("reason", "") for _, info in format_rows]
            df_out["format_char_ratio"] = [info.get("char_ratio", np.nan) for _, info in format_rows]
            df_out["format_impression_markers"] = [info.get("impression_markers", np.nan) for _, info in format_rows]
        else:
            df_out["format_ok"] = True

    def _ok_col(name: str, default: bool) -> pd.Series:
        if name in df_out.columns:
            return df_out[name].fillna(False).astype(bool)
        return pd.Series(default, index=index)

    quick_ok = pd.Series(True, index=index) if not USE_QUICK_SAFETY else _ok_col("quick_ok", False)
    lex_ok = pd.Series(True, index=index) if not USE_LEXICAL_FILTER else _ok_col("lex_ok", False)
    chex_ok = pd.Series(True, index=index) if not USE_CHEXBERT else _ok_col("chex_ok", False)
    rg_ok = pd.Series(True, index=index) if not USE_RADGRAPH else _ok_col("rg_ok", False)
    msl_ok = pd.Series(True, index=index) if not USE_MEDSIGLIP else _ok_col("msl_ok", medsiglip is None)
    ascii_ok = pd.Series(True, index=index) if not USE_ASCII_FILTER else _ok_col("ascii_ok", False)
    format_ok = pd.Series(True, index=index) if not USE_FORMAT_FILTER else _ok_col("format_ok", False)

    df_out["keep_ok"] = quick_ok & lex_ok & chex_ok & rg_ok & msl_ok & ascii_ok & format_ok
    df_out["all_filters_pass"] = df_out["keep_ok"]
    return df_out


def select_best_candidates(df_cands: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df_cands) == 0:
        return df_cands, df_cands

    df_cands = apply_selection_gates(df_cands)
    df_pass = df_cands[df_cands["keep_ok"] == True].copy()
    if len(df_pass) == 0:
        return df_cands, df_pass

    if "msl_sim" in df_pass.columns and df_pass["msl_sim"].notna().any():
        df_pass = df_pass.sort_values(["ex_id", "msl_sim", "cand_id"], ascending=[True, False, True])
    else:
        df_pass = df_pass.sort_values(["ex_id", "cand_id"], ascending=[True, True])
    df_best = df_pass.groupby("ex_id", as_index=False).head(1).reset_index(drop=True)
    return df_cands, df_best


def run_paraphrase_with_filters(
    df_in: pd.DataFrame,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every: int = 200,   # в примерах, не в кандидатах
    run_tag: str = "e2c",
    resume: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    t0 = time.time()

    def _empty_filter(reason: str = "") -> Dict[str, Any]:
        return {
            "quick_ok": False,
            "lex_ok": False,
            "chex_ok": False,
            "rg_ok": False,
            "msl_ok": False,
            "msl_sim": np.nan,
            "ascii_ok": False,
            "format_ok": False,
            "keep_ok": False,
            "all_filters_pass": False,
            "filter_error": reason,
        }

    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    done_ex_ids: set[str] = set()
    part_idx = 0
    part_re = re.compile(rf"e2c_candidates_{re.escape(run_tag)}_part(\d+)\.parquet$")

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        existing_parts = sorted(checkpoint_dir.glob(f"e2c_candidates_{run_tag}_part*.parquet"))

        if existing_parts and not resume:
            raise FileExistsError(
                f"resume=False but existing E2C chunks were found in {checkpoint_dir}. "
                "Use a fresh data.e2c_out_dir or move old chunks before a clean regeneration."
            )

        if existing_parts:
            idxs = []
            for p in existing_parts:
                m = part_re.search(p.name)
                if m:
                    idxs.append(int(m.group(1)))
            part_idx = (max(idxs) + 1) if idxs else len(existing_parts)

        if resume and existing_parts:
            for p in tqdm(existing_parts, desc="Load existing E2c chunks"):
                df_prev = pd.read_parquet(p, columns=["ex_id"]) if p.exists() else pd.DataFrame()
                if "ex_id" in df_prev.columns:
                    done_ex_ids.update(df_prev["ex_id"].astype(str).tolist())
            print(f"[CKPT] resume=True, chunks={len(existing_parts)}, done_examples={len(done_ex_ids)}")

    rows_buf: List[Dict[str, Any]] = []
    processed_examples = 0
    last_flush_examples = 0

    def _flush(reason: str) -> None:
        nonlocal rows_buf, part_idx
        if checkpoint_dir is None or len(rows_buf) == 0:
            return
        df_part = pd.DataFrame(rows_buf)
        p = checkpoint_dir / f"e2c_candidates_{run_tag}_part{part_idx:05d}.parquet"
        df_part.to_parquet(p, index=False, engine="pyarrow", compression="snappy")
        print(f"[CKPT] {reason}: saved {len(df_part)} candidate rows -> {p}")
        rows_buf = []
        part_idx += 1

    def _append_generate_failed(r: pd.Series, src_full: str, ex_id: str, err: Exception) -> None:
        rows_buf.append({
            "model_id": paraphraser.model_name,
            "ex_id": ex_id,
            "cand_id": -1,
            "raw": "",
            "parsed_ok": False,
            "aug_findings": "",
            "aug_impression": "",
            "cand_full": "",
            "src_full": src_full,
            "t_gen_s": 0.0,
            "subject_id": int(r["subject_id"]),
            "study_id": int(r["study_id"]),
            "dicom_id": str(r["dicom_id"]),
            "dcm_path": str(r["dcm_path"]),
            "split": str(r["split"]),
            "shard_id": int(r["shard_id"]),
            **_empty_filter(f"generate_failed: {repr(err)}"),
        })

    def _process_batch(batch_rows: List[pd.Series]) -> None:
        nonlocal processed_examples, last_flush_examples
        prompts: List[str] = []
        src_fulls: List[str] = []
        ex_ids: List[str] = []

        for r in batch_rows:
            f0 = str(r["ref_findings"])
            i0 = str(r["ref_impression"])
            src_full = build_target_text(f0, i0)
            prompts.append(build_paraphrase_prompt(f0, i0))
            src_fulls.append(src_full)
            ex_ids.append(f"{int(r['study_id'])}")

        try:
            t_gen0 = time.time()
            outs_by_prompt = paraphraser.generate(prompts)
            t_gen = (time.time() - t_gen0) / max(1, len(batch_rows))
        except Exception as e:
            if len(batch_rows) > 1:
                for r in batch_rows:
                    _process_batch([r])
                return
            _append_generate_failed(batch_rows[0], src_fulls[0], ex_ids[0], e)
            processed_examples += 1
            return

        for r, ex_id, src_full, outs in zip(batch_rows, ex_ids, src_fulls, outs_by_prompt):
            for k, raw in enumerate(outs):
                pf, pi = parse_two_lines(raw)
                parsed_ok = bool(pf and pi)
                cand_full = build_target_text(pf, pi) if parsed_ok else ""

                base = {
                    "model_id": paraphraser.model_name,
                    "ex_id": ex_id,
                    "cand_id": k,
                    "raw": raw,
                    "parsed_ok": parsed_ok,
                    "aug_findings": pf,
                    "aug_impression": pi,
                    "cand_full": cand_full,
                    "src_full": src_full,
                    "t_gen_s": float(t_gen),
                    "subject_id": int(r["subject_id"]),
                    "study_id": int(r["study_id"]),
                    "dicom_id": str(r["dicom_id"]),
                    "dcm_path": str(r["dcm_path"]),
                    "split": str(r["split"]),
                    "shard_id": int(r["shard_id"]),
                }

                if parsed_ok:
                    try:
                        filt = eval_candidate_filters(src_full, cand_full)
                        filt.setdefault("filter_error", "")
                    except Exception as e:
                        filt = _empty_filter(f"filter_failed: {repr(e)}")
                else:
                    filt = _empty_filter("parse_failed")

                rows_buf.append({**base, **filt})

        processed_examples += len(batch_rows)
        if checkpoint_dir is not None and checkpoint_every > 0 and processed_examples - last_flush_examples >= checkpoint_every:
            _flush("periodic")
            last_flush_examples = processed_examples

    batch_rows: List[pd.Series] = []
    outer_batch_size = max(1, int(globals().get("LLM_BATCH_SIZE", 1)))

    for _, r in tqdm(df_in.iterrows(), total=len(df_in), desc="E2c paraphrase"):
        ex_id = f"{int(r['study_id'])}"

        if resume and ex_id in done_ex_ids:
            continue

        batch_rows.append(r)
        if len(batch_rows) >= outer_batch_size:
            _process_batch(batch_rows)
            batch_rows = []

    if batch_rows:
        _process_batch(batch_rows)

    _flush("final")

    if checkpoint_dir is not None:
        part_paths = sorted(checkpoint_dir.glob(f"e2c_candidates_{run_tag}_part*.parquet"))
        if len(part_paths) > 0:
            df_cands = pd.concat([pd.read_parquet(p) for p in part_paths], ignore_index=True)
        else:
            df_cands = pd.DataFrame(rows_buf)
    else:
        df_cands = pd.DataFrame(rows_buf)

    if len(df_cands) == 0:
        df_pass = df_cands
        df_best = df_cands
    else:
        df_cands, df_best = select_best_candidates(df_cands)
        df_pass = df_cands[df_cands["keep_ok"] == True].copy()

    n_examples = int(df_in["study_id"].nunique()) if "study_id" in df_in.columns else len(df_in)
    print("Candidates:", len(df_cands), "Passing (keep_ok):", len(df_pass), "Examples:", n_examples)
    print("Time total (s):", time.time() - t0)
    return df_cands, df_best


__all__ = [
    # config/globals
    "SEED",
    "AUX_DEVICE",
    "USE_QUICK_SAFETY",
    "USE_LEXICAL_FILTER",
    "USE_ASCII_FILTER",
    "USE_FORMAT_FILTER",
    "USE_RADGRAPH",
    "USE_CHEXBERT",
    "USE_MEDSIGLIP",
    "LEX_MAX_SEQ_RATIO",
    "LEX_MAX_4GRAM_OVERLAP",
    "LEX_MIN_TOKEN_CHANGE",
    "FORMAT_MIN_CHAR_RATIO",
    "FORMAT_MAX_CHAR_RATIO",
    "MIN_ENT_P",
    "MIN_ENT_R",
    "MIN_REL_P",
    "MIN_REL_R",
    "MAX_EXTRA_ENT",
    "MAX_MISSING_ENT",
    "MAX_EXTRA_REL",
    "MAX_MISSING_REL",
    "RADGRAPH_MODEL_TYPE",
    "MEDSIGLIP_MODEL_ID",
    "MEDSIGLIP_MODE",
    "MEDSIGLIP_MIN_SIM",
    "MEDSIGLIP_MAX_SIM",
    "LLM_BATCH_SIZE",
    "LLM_DEVICE",
    "LLM_GPU_IDS_LIST",
    "LLM_NUM_GPUS",
    # globals objects
    "rg",
    "medsiglip",
    "chexbert_runner",
    "paraphraser",
    # funcs/classes used by prepare.py
    "add_shard_id",
    "save_all_shards",
    "ModelSpec",
    "HFParaphraser",
    "try_init_radgraph",
    "try_init_medsiglip",
    "CheXbertRunner",
    "run_paraphrase_with_filters",
    # extra helpers
    "build_target_text",
    "build_paraphrase_prompt",
    "parse_two_lines",
    "format_safety_ok",
    "apply_selection_gates",
    "select_best_candidates",
    "eval_candidate_filters",
]