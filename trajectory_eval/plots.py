"""Figures: the headline forest plot (overlapping CIs) and the Lorenz curve of error mass.

Imports matplotlib lazily so the rest of the toolkit works in headless/minimal environments.
"""
from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np


def forest(results: Mapping[str, Tuple[float, float, float]], xlabel: str = "metric",
           title: str = "", sort: bool = True, savepath: str = None):
    """Forest plot of (point, lo, hi) per model. If CIs overlap, the 'improvement' is within
    noise. `results` maps label -> (point, ci_lo, ci_hi)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = list(results.items())
    if sort:
        items.sort(key=lambda kv: kv[1][0], reverse=True)
    labels = [k for k, _ in items]
    pts = np.array([v[0] for _, v in items])
    lo = np.array([v[1] for _, v in items])
    hi = np.array([v[2] for _, v in items])
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 0.6 * len(labels) + 1.5))
    ax.errorbar(pts, y, xerr=[pts - lo, hi - pts], fmt="o", capsize=4, color="#1f4e79")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig


def lorenz(series: Mapping[str, np.ndarray], savepath: str = None, title: str = "Error concentration"):
    """Lorenz-style curve: cumulative share of total error vs fraction of tracks (worst-first).
    A sharply bent curve = a few tracks own most of the error (the heavy tail)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, x in series.items():
        x = np.sort(np.asarray(x, float).ravel())[::-1]   # worst first
        cum = np.cumsum(x) / x.sum()
        frac = np.arange(1, x.size + 1) / x.size
        ax.plot(frac, cum, label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="uniform (no tail)")
    ax.set_xlabel("fraction of tracks (worst first)")
    ax.set_ylabel("cumulative share of total error")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
    return fig
