from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def _cfg_bool(cfg: Mapping[str, Any], key: str, default: bool) -> bool:
    raw = cfg.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(raw)


def _collect_prefixed(payload: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k.startswith(prefix)}


def _to_jsonable(x: Any) -> Any:
    if isinstance(x, Mapping):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [_to_jsonable(v) for v in x]
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return x.item()
        except Exception:
            return x
    return x


_SPACE_RE = re.compile(r"\s+")
_ESCAPED_PUNCT_RE = re.compile(r"\\([^\w\s])")


def _sanitize_ratescore_text(text: str) -> str:
    s = str(text or "").replace("\r", " ").replace("\n", " ")
    # Typical problematic fragments for RaTEScore parser: \. \, \( \) etc.
    s = _ESCAPED_PUNCT_RE.sub(r"\1", s)
    # Remove remaining literal backslashes that often come from over-escaping.
    s = s.replace("\\", " ")
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def _sanitize_ratescore_inputs(preds: list[str], refs: list[str]) -> tuple[list[str], list[str], dict[str, int]]:
    safe_preds = [_sanitize_ratescore_text(x) for x in preds]
    safe_refs = [_sanitize_ratescore_text(x) for x in refs]
    stats = {
        "preds_changed": int(sum(1 for a, b in zip(preds, safe_preds) if str(a or "") != b)),
        "refs_changed": int(sum(1 for a, b in zip(refs, safe_refs) if str(a or "") != b)),
    }
    return safe_preds, safe_refs, stats


def _run_radeval(RadEval, flags: dict[str, Any], refs: list[str], preds: list[str]) -> dict[str, Any]:
    evaluator = RadEval(**flags)
    return _to_jsonable(evaluator(refs=refs, hyps=preds))


def compute_radeval_metrics(preds: list[str], refs: list[str], cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = cfg or {}
    enabled = _cfg_bool(cfg, "enabled", False)
    if not enabled:
        return {
            "enabled": False,
            "available": False,
            "reason": "disabled_by_config",
            "selected_metrics": {},
            "all_metrics": {},
        }

    try:
        from RadEval import RadEval
    except Exception as e:  # pragma: no cover
        return {
            "enabled": True,
            "available": False,
            "reason": "import_error",
            "error": repr(e),
            "selected_metrics": {},
            "all_metrics": {},
        }

    flags = {
        "do_bertscore": _cfg_bool(cfg, "do_bertscore", True),
        "do_f1chexbert": _cfg_bool(cfg, "do_f1chexbert", True),
        "do_ratescore": _cfg_bool(cfg, "do_ratescore", True),
        "do_radcliq": _cfg_bool(cfg, "do_radcliq", True),
        "do_green": _cfg_bool(cfg, "do_green", True),
        "do_radgraph": _cfg_bool(cfg, "do_radgraph", True),
        "do_details": _cfg_bool(cfg, "do_details", False),
        "do_per_sample": _cfg_bool(cfg, "do_per_sample", False),
    }

    sanitize_ratescore_input = _cfg_bool(cfg, "sanitize_ratescore_input", True)
    metric_flags = ["do_bertscore", "do_f1chexbert", "do_ratescore", "do_radcliq", "do_green", "do_radgraph"]

    warnings: list[dict[str, str]] = []
    all_metrics: dict[str, Any] = {}
    ratescore_sanitization: dict[str, int] | None = None

    if flags["do_ratescore"] and sanitize_ratescore_input:
        base_flags = dict(flags)
        base_flags["do_ratescore"] = False

        if any(bool(base_flags[k]) for k in metric_flags):
            try:
                all_metrics.update(_run_radeval(RadEval, base_flags, refs=refs, preds=preds))
            except Exception as e:  # pragma: no cover
                warnings.append({"stage": "base_metrics", "error": repr(e)})

        ratescore_flags = {k: False for k in flags}
        ratescore_flags["do_ratescore"] = True
        ratescore_flags["do_details"] = flags["do_details"]
        ratescore_flags["do_per_sample"] = flags["do_per_sample"]

        safe_preds, safe_refs, ratescore_sanitization = _sanitize_ratescore_inputs(preds, refs)
        try:
            all_metrics.update(_run_radeval(RadEval, ratescore_flags, refs=safe_refs, preds=safe_preds))
        except Exception as e:  # pragma: no cover
            warnings.append({"stage": "ratescore", "error": repr(e)})
    else:
        try:
            all_metrics = _run_radeval(RadEval, flags, refs=refs, preds=preds)
        except Exception as e:  # pragma: no cover
            return {
                "enabled": True,
                "available": False,
                "reason": "execution_error",
                "error": repr(e),
                "flags": flags,
                "selected_metrics": {},
                "all_metrics": {},
            }

    if not all_metrics and warnings:
        return {
            "enabled": True,
            "available": False,
            "reason": "execution_error",
            "flags": flags,
            "warnings": warnings,
            "selected_metrics": {},
            "all_metrics": {},
        }

    selected = {
        "bertscore": all_metrics.get("bertscore"),
        "ratescore": all_metrics.get("ratescore"),
        "radcliq_v1": all_metrics.get("radcliq_v1"),
        "green": all_metrics.get("green"),
        "radgraph_simple": all_metrics.get("radgraph_simple"),
        "radgraph_partial": all_metrics.get("radgraph_partial"),
        "radgraph_complete": all_metrics.get("radgraph_complete"),
        "f1chexbert": _collect_prefixed(all_metrics, "f1chexbert_"),
    }

    return {
        "enabled": True,
        "available": True,
        "flags": flags,
        "warnings": warnings,
        "ratescore_input_sanitization": ratescore_sanitization,
        "selected_metrics": selected,
        "all_metrics": all_metrics,
    }
