"""Statistical rigor for trajectory-prediction evaluation -- the core of Experiment A.

What the field does:   report one ADE/FDE point estimate per method, no error bars, no test.
What this module adds:  bootstrap confidence intervals, seed variance, and proper
                        paired/unpaired significance tests with effect sizes, so we can ask
                        "is this 'improvement' distinguishable from noise?"

p-values come from established tests (Mann-Whitney U unpaired, Wilcoxon signed-rank paired).
The bootstrap is used for confidence intervals on the statistic (its rigorous use). A
percentile bootstrap two-sided p (achieved significance level) is also reported for the
difference, clearly labelled as approximate. Multiple comparisons -> Holm-Bonferroni.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats


# ---- reducers (operate on a 2-D bootstrap array, axis=1) ----------------------------------
def r_mean(a: np.ndarray) -> np.ndarray:
    return a.mean(axis=1)


def r_cvar(alpha: float = 0.95) -> Callable[[np.ndarray], np.ndarray]:
    """Reducer: CVaR_alpha = mean of the worst (1-alpha) fraction (higher value = worse)."""
    def f(a: np.ndarray) -> np.ndarray:
        n = a.shape[1]
        # -1e-9 guards against float overshoot, e.g. (1-0.95)*100 = 5.0000000000004 -> ceil 6
        k = max(1, int(np.ceil((1.0 - alpha) * n - 1e-9)))
        # k largest along axis=1 without full sort
        part = np.partition(a, n - k, axis=1)[:, n - k:]
        return part.mean(axis=1)
    return f


# ---- bootstrap ----------------------------------------------------------------------------
def bootstrap_ci(x: np.ndarray, reducer: Callable[[np.ndarray], np.ndarray] = r_mean,
                 n_boot: int = 10000, alpha: float = 0.05, seed: int = 0,
                 batch: int = 1000):
    """Percentile bootstrap CI for `reducer` of x.

    Returns (point, lo, hi) at the (1-alpha) confidence level.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        raise ValueError("empty input")
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        b = min(batch, n_boot - done)
        idx = rng.integers(0, n, size=(b, n), dtype=np.int32)
        reps[done:done + b] = reducer(x[idx])
        done += b
    point = float(reducer(x[None, :])[0])
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def cliffs_delta_from_u(u: float, n1: int, n2: int) -> float:
    """Cliff's delta from the Mann-Whitney U of the FIRST sample.
    +1 => first sample's values are systematically larger (here: larger error = worse)."""
    return 2.0 * u / (n1 * n2) - 1.0


def _boot_diff_ci(a, b, n_boot, alpha, seed, paired, batch=1000):
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=np.float64)
    na, nb = a.size, b.size
    done = 0
    while done < n_boot:
        nb_ = min(batch, n_boot - done)
        if paired:
            idx = rng.integers(0, na, size=(nb_, na), dtype=np.int32)
            reps[done:done + nb_] = (a[idx] - b[idx]).mean(axis=1)
        else:
            ia = rng.integers(0, na, size=(nb_, na), dtype=np.int32)
            ib = rng.integers(0, nb, size=(nb_, nb), dtype=np.int32)
            reps[done:done + nb_] = a[ia].mean(axis=1) - b[ib].mean(axis=1)
        done += nb_
    lo, hi = np.percentile(reps, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided percentile-bootstrap achieved significance level
    p = 2.0 * min((reps <= 0).mean(), (reps >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.0))


def compare_unpaired(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Unpaired comparison of two error samples (different tracks / different lengths OK).

    Reports difference of means with bootstrap CI, Mann-Whitney U p-value, and Cliff's delta.
    `diff` = mean(a) - mean(b); positive => a is worse (larger error).
    """
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    u, p_mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
    lo, hi, p_boot = _boot_diff_ci(a, b, n_boot, alpha, seed, paired=False)
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "n_a": int(a.size), "n_b": int(b.size),
        "diff": float(a.mean() - b.mean()), "diff_ci": (lo, hi),
        "mwu_p": float(p_mwu), "boot_p": p_boot,
        "cliffs_delta": cliffs_delta_from_u(float(u), a.size, b.size),
    }


def compare_paired(a, b, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Paired comparison (same tracks, equal length). Wilcoxon signed-rank + bootstrap CI."""
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size != b.size:
        raise ValueError(f"paired compare needs equal length; got {a.size} vs {b.size}")
    d = a - b
    # Wilcoxon errors if all differences are zero; guard it.
    if np.allclose(d, 0):
        p_w = 1.0
    else:
        p_w = float(stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    lo, hi, p_boot = _boot_diff_ci(a, b, n_boot, alpha, seed, paired=True)
    return {
        "mean_a": float(a.mean()), "mean_b": float(b.mean()), "n": int(a.size),
        "diff": float(d.mean()), "diff_ci": (lo, hi),
        "median_diff": float(np.median(d)), "frac_a_worse": float((d > 0).mean()),
        "wilcoxon_p": p_w, "boot_p": p_boot,
    }


def holm_bonferroni(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (monotone, clipped to 1)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def seed_summary(values: Sequence[float]) -> dict:
    """Aggregate a scalar metric across training seeds (Exp A, layer 1)."""
    v = np.asarray(values, dtype=np.float64)
    return {
        "n_seeds": int(v.size), "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()), "max": float(v.max()),
    }
