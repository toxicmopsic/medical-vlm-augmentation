from __future__ import annotations

from statistics import mean



def safe_mean(values):
    vals = list(values)
    return float(mean(vals)) if vals else 0.0
