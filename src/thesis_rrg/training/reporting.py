from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _normalize_report_to(raw: Any) -> str | list[str]:
    if raw is None:
        return "none"

    if isinstance(raw, str):
        token = raw.strip()
        if not token:
            return "none"
        if "," in token:
            items = [x.strip() for x in token.split(",") if x.strip()]
            return items or "none"
        return token

    if isinstance(raw, Sequence):
        items = [str(x).strip() for x in raw if str(x).strip()]
        return items or "none"

    return str(raw)


def resolve_report_to(cfg) -> str | list[str]:
    training_report_to = cfg.training.get("report_to", None)
    if training_report_to is not None:
        return _normalize_report_to(training_report_to)

    logging_report_to = cfg.logging.get("report_to", "none")
    return _normalize_report_to(logging_report_to)


def report_to_includes_tensorboard(report_to: str | list[str]) -> bool:
    if isinstance(report_to, str):
        return report_to.strip().lower() in {"tensorboard", "tb"}
    return any(str(x).strip().lower() in {"tensorboard", "tb"} for x in report_to)


def ensure_reporting_dependencies(report_to: str | list[str]) -> None:
    if report_to_includes_tensorboard(report_to):
        try:
            import tensorboard  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "TensorBoard logging selected but `tensorboard` is not installed. "
                "Install it with `pip install tensorboard`."
            ) from e
