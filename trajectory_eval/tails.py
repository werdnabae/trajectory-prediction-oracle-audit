"""Heavy-tail characterization -- Experiment C.

The aggregate mean ADE/FDE that the field reports is dominated by a rare, hard tail of
tracks. This module quantifies that tail and tests whether the leaderboard *ranking* even
survives when you score the tail instead of the mean.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy import stats as _sps


def cvar(x: np.ndarray, alpha: float = 0.95) -> float:
    """CVaR_alpha: mean of the worst (1-alpha) fraction of errors (higher = worse)."""
    x = np.asarray(x, float).ravel()
    n = x.size
    k = max(1, int(np.ceil((1.0 - alpha) * n - 1e-9)))   # -1e-9: float overshoot guard
    return float(np.partition(x, n - k)[n - k:].mean())


def tail_share(x: np.ndarray, frac: float = 0.05) -> float:
    """Fraction of the SUMMED error contributed by the worst `frac` of tracks."""
    x = np.asarray(x, float).ravel()
    n = x.size
    k = max(1, int(np.ceil(frac * n - 1e-9)))   # -1e-9: float overshoot guard
    total = x.sum()
    if total <= 0:
        return float("nan")
    return float(np.partition(x, n - k)[n - k:].sum() / total)


def gini(x: np.ndarray) -> float:
    """Gini coefficient of the error magnitudes (0 = uniform, ->1 = concentrated in few tracks)."""
    x = np.sort(np.asarray(x, float).ravel())
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2.0 * (idx * x).sum()) / (n * x.sum()) - (n + 1.0) / n)


def summarize(x: np.ndarray) -> dict:
    """Full distributional summary of a per-track error array."""
    x = np.asarray(x, float).ravel()
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
        "mean_over_median": float(x.mean() / np.median(x)) if np.median(x) > 0 else float("nan"),
        "skewness": float(_sps.skew(x)),
        "excess_kurtosis": float(_sps.kurtosis(x)),   # 0 for a normal distribution
        "cvar95": cvar(x, 0.95),
        "tail_share_top5pct": tail_share(x, 0.05),
        "gini": gini(x),
    }


def rank_flip(metrics: Mapping[str, Mapping[str, float]], key_a: str, key_b: str) -> dict:
    """Does the model ranking change between two metrics (lower = better for both)?

    `metrics` maps model -> {metric_name: value}. Returns the two orderings and the
    Kendall-tau / Spearman correlation between the induced rankings.
    """
    models = list(metrics)
    va = np.array([metrics[m][key_a] for m in models])
    vb = np.array([metrics[m][key_b] for m in models])
    order_a = [models[i] for i in np.argsort(va)]
    order_b = [models[i] for i in np.argsort(vb)]
    tau = _sps.kendalltau(va, vb).correlation if len(models) > 1 else float("nan")
    rho = _sps.spearmanr(va, vb).correlation if len(models) > 2 else float("nan")
    return {
        "order_by_" + key_a: order_a,
        "order_by_" + key_b: order_b,
        "kendall_tau": float(tau) if tau is not None else float("nan"),
        "spearman_rho": float(rho) if rho is not None else float("nan"),
        "ranking_changed": order_a != order_b,
    }
