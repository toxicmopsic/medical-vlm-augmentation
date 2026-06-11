from __future__ import annotations

import random
from typing import Callable

import numpy as np



def bootstrap_ci(
    values: list[float],
    statistic_fn: Callable[[list[float]], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    rng = random.Random(seed)
    n = len(values)
    stats = []

    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(float(statistic_fn(sample)))

    stats_arr = np.array(stats)
    low = float(np.quantile(stats_arr, alpha / 2))
    high = float(np.quantile(stats_arr, 1 - alpha / 2))

    return {
        "mean": float(statistic_fn(values)),
        "ci_low": low,
        "ci_high": high,
        "n": n,
    }
