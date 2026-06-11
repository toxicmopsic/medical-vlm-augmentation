from __future__ import annotations

from typing import Any

from thesis_rrg.metrics.radeval_metrics import compute_radeval_metrics
from thesis_rrg.metrics.radgraph_metrics import compute_radgraph_detail, compute_radgraph_f1
from thesis_rrg.metrics.text_metrics import compute_text_metrics



def run_reportgen_suite(
    preds_findings: list[str],
    refs_findings: list[str],
    preds_impression: list[str],
    refs_impression: list[str],
    preds_full: list[str],
    refs_full: list[str],
    radgraph_model_type: str,
    radgraph_reward_level: str,
    radeval_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text_findings = compute_text_metrics(preds_findings, refs_findings)
    text_impression = compute_text_metrics(preds_impression, refs_impression)
    text_full = compute_text_metrics(preds_full, refs_full)

    rg_findings = compute_radgraph_f1(
        preds_findings,
        refs_findings,
        model_type=radgraph_model_type,
        reward_level=radgraph_reward_level,
    )
    rg_impression = compute_radgraph_f1(
        preds_impression,
        refs_impression,
        model_type=radgraph_model_type,
        reward_level=radgraph_reward_level,
    )
    rg_full = compute_radgraph_f1(
        preds_full,
        refs_full,
        model_type=radgraph_model_type,
        reward_level=radgraph_reward_level,
    )

    rg_detail_findings = compute_radgraph_detail(
        preds_findings,
        refs_findings,
        model_type=radgraph_model_type,
        reward_level=radgraph_reward_level,
    )
    rg_detail_impression = compute_radgraph_detail(
        preds_impression,
        refs_impression,
        model_type=radgraph_model_type,
        reward_level=radgraph_reward_level,
    )
    rg_detail_full = compute_radgraph_detail(
        preds_full,
        refs_full,
        model_type=radgraph_model_type,
        reward_level=radgraph_reward_level,
    )

    radeval_full = compute_radeval_metrics(
        preds=preds_full,
        refs=refs_full,
        cfg=radeval_cfg or {},
    )

    # Compare RadGraph from two independent implementations on full reports.
    radgraph_compare_full: dict[str, Any] = {
        "available": False,
        "reason": "insufficient_inputs",
    }
    radeval_partial = None
    if radeval_full.get("available", False):
        radeval_partial = (radeval_full.get("selected_metrics", {}) or {}).get("radgraph_partial")
    legacy_partial = rg_full.get("rg_er") if isinstance(rg_full, dict) else None
    if isinstance(radeval_partial, (float, int)) and isinstance(legacy_partial, (float, int)):
        radgraph_compare_full = {
            "available": True,
            "radeval_radgraph_partial": float(radeval_partial),
            "legacy_radgraph_rg_er": float(legacy_partial),
            "absolute_delta": float(abs(float(radeval_partial) - float(legacy_partial))),
        }

    return {
        "text_metrics_findings": text_findings,
        "text_metrics_impression": text_impression,
        "text_metrics_full": text_full,
        "radgraph_findings": rg_findings,
        "radgraph_impression": rg_impression,
        "radgraph_full": rg_full,
        "radgraph_detail_findings": rg_detail_findings,
        "radgraph_detail_impression": rg_detail_impression,
        "radgraph_detail_full": rg_detail_full,
        "radeval_full": radeval_full,
        "radgraph_compare_full": radgraph_compare_full,
        # Explicit aliases for clarity in analysis scripts.
        "radgraph_legacy_findings": rg_findings,
        "radgraph_legacy_impression": rg_impression,
        "radgraph_legacy_full": rg_full,
        "radgraph_radeval_full": (
            radeval_full.get("selected_metrics", {})
            if isinstance(radeval_full, dict)
            else {}
        ),
    }
