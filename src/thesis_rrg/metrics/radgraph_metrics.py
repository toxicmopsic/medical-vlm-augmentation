from __future__ import annotations

import re
from typing import Any

from tqdm.auto import tqdm

_SPACE_RE = re.compile(r"\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _simple_token_count(text: str) -> int:
    clean = _NON_ALNUM_RE.sub(" ", text.lower()).strip()
    if not clean:
        return 0
    return len([t for t in clean.split(" ") if t])


def sanitize_for_radgraph(text: str) -> str:
    """
    Normalize free-text report so RadGraph does not receive empty/one-token
    sentences that trigger warnings and unstable parsing behavior.
    """
    raw = str(text or "").replace("\r", " ").replace("\n", " ")
    raw = _SPACE_RE.sub(" ", raw).strip()
    if not raw:
        return "no findings."

    parts = _SENT_SPLIT_RE.split(raw)
    kept: list[str] = []
    for part in parts:
        p = part.strip(" ;,")
        if not p:
            continue
        if _simple_token_count(p) >= 2:
            kept.append(p)

    if kept:
        out = ". ".join(kept).strip()
    else:
        # Fallback: keep content but guarantee at least two lexical tokens.
        toks = [t for t in _NON_ALNUM_RE.sub(" ", raw.lower()).split(" ") if t]
        if len(toks) >= 2:
            out = " ".join(toks)
        elif len(toks) == 1:
            out = f"{toks[0]} {toks[0]}"
        else:
            out = "no findings"

    if not out.endswith((".", "!", "?")):
        out = out + "."
    return out


def _prepare_radgraph_inputs(preds: list[str], refs: list[str]) -> tuple[list[str], list[str], dict[str, int]]:
    safe_preds = [sanitize_for_radgraph(x) for x in preds]
    safe_refs = [sanitize_for_radgraph(x) for x in refs]
    stats = {
        "preds_changed": int(sum(1 for a, b in zip(preds, safe_preds) if str(a or "") != b)),
        "refs_changed": int(sum(1 for a, b in zip(refs, safe_refs) if str(a or "") != b)),
    }
    return safe_preds, safe_refs, stats


def compute_radgraph_f1(preds: list[str], refs: list[str], model_type: str, reward_level: str) -> dict[str, Any]:
    try:
        from radgraph import F1RadGraph
    except Exception as e:  # pragma: no cover
        return {
            "available": False,
            "error": f"radgraph is unavailable: {e}",
            "n": len(preds),
        }

    safe_preds, safe_refs, sanitize_stats = _prepare_radgraph_inputs(preds, refs)

    f1rg = F1RadGraph(reward_level=reward_level, model_type=model_type)
    mean_reward, _, _, _ = f1rg(hyps=safe_preds, refs=safe_refs)
    rg_e, rg_er, rg_bar_er = mean_reward

    return {
        "available": True,
        "model_type": model_type,
        "n": len(preds),
        "rg_e": float(rg_e),
        "rg_er": float(rg_er),
        "rg_bar_er": float(rg_bar_er),
        "input_sanitization": sanitize_stats,
    }


_TOK_CLEAN_RE = re.compile(r"[^a-z0-9]+")


def _clean_tok(x: str) -> str:
    return _TOK_CLEAN_RE.sub("", str(x).lower().strip())


def _norm_tokens(tok: Any) -> tuple[str, ...]:
    if tok is None:
        return tuple()
    if isinstance(tok, list):
        out = [_clean_tok(t) for t in tok]
        return tuple(t for t in out if t)
    if isinstance(tok, str):
        out = [_clean_tok(p) for p in tok.split(" ")]
        return tuple(t for t in out if t)
    t = _clean_tok(tok)
    return (t,) if t else tuple()


def _unwrap_radgraph_payload(ann_dict: Any) -> dict[str, Any] | None:
    if not isinstance(ann_dict, dict) or len(ann_dict) == 0:
        return None

    payload: Any = ann_dict[next(iter(ann_dict.keys()))]
    for _ in range(6):
        if isinstance(payload, dict) and "entities" in payload:
            return payload
        if isinstance(payload, dict) and len(payload) == 1:
            payload = next(iter(payload.values()))
            continue
        if isinstance(payload, dict) and "0" in payload and isinstance(payload["0"], dict):
            payload = payload["0"]
            continue
        break
    return payload if (isinstance(payload, dict) and "entities" in payload) else None


def _rg_to_sets(ann_dict: Any) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]]]:
    payload = _unwrap_radgraph_payload(ann_dict)
    if payload is None:
        return set(), set()

    entities = payload.get("entities", {}) or {}
    ent_map: dict[str, tuple[Any, ...]] = {}
    ent_set: set[tuple[Any, ...]] = set()

    for ent_id, ent in entities.items():
        lab = str((ent or {}).get("label", "")).strip()
        tok = _norm_tokens((ent or {}).get("tokens", []))
        key = (tok, lab)
        ent_map[str(ent_id)] = key
        ent_set.add(key)

    rel_set: set[tuple[Any, ...]] = set()
    for src_id, ent in entities.items():
        src_key = ent_map.get(str(src_id))
        if src_key is None:
            continue
        rels = (ent or {}).get("relations", []) or []
        for rel in rels:
            if not isinstance(rel, (list, tuple)) or len(rel) != 2:
                continue
            rel_type, tgt_id = str(rel[0]), str(rel[1])
            tgt_key = ent_map.get(str(tgt_id))
            if tgt_key is None:
                continue
            rel_set.add((src_key, rel_type, tgt_key))

    return ent_set, rel_set


def _pr(ref_set: set[Any], hyp_set: set[Any]) -> tuple[float, float]:
    if len(ref_set) == 0 and len(hyp_set) == 0:
        return 1.0, 1.0
    if len(hyp_set) == 0:
        return 0.0, 0.0
    inter = len(ref_set & hyp_set)
    p = inter / max(1, len(hyp_set))
    r = inter / max(1, len(ref_set))
    return float(p), float(r)



def compute_radgraph_detail(preds: list[str], refs: list[str], model_type: str, reward_level: str) -> dict[str, Any]:
    try:
        from radgraph import RadGraph
    except Exception as e:  # pragma: no cover
        return {
            "available": False,
            "error": f"radgraph is unavailable: {e}",
            "n": len(preds),
        }

    import numpy as np

    safe_preds, safe_refs, sanitize_stats = _prepare_radgraph_inputs(preds, refs)

    rg = RadGraph(reward_level=reward_level, model_type=model_type)
    ent_p, ent_r, rel_p, rel_r = [], [], [], []
    skipped = 0

    for hyp, ref in tqdm(
        zip(safe_preds, safe_refs),
        total=len(safe_preds),
        desc="RadGraph detail",
        dynamic_ncols=True,
    ):
        try:
            hyp_anns = rg([hyp])
            ref_anns = rg([ref])
            hyp_ent, hyp_rel = _rg_to_sets(hyp_anns)
            ref_ent, ref_rel = _rg_to_sets(ref_anns)

            p_e, r_e = _pr(ref_ent, hyp_ent)
            p_r, r_r = _pr(ref_rel, hyp_rel)
            ent_p.append(p_e)
            ent_r.append(r_e)
            rel_p.append(p_r)
            rel_r.append(r_r)
        except Exception:
            skipped += 1
            continue

    def _mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    return {
        "available": True,
        "method": "set_overlap_v2",
        "n": len(preds),
        "n_scored": int(len(ent_p)),
        "n_skipped": int(skipped),
        "input_sanitization": sanitize_stats,
        "entity_precision": _mean(ent_p),
        "entity_recall": _mean(ent_r),
        "relation_precision": _mean(rel_p),
        "relation_recall": _mean(rel_r),
    }
