"""Safety-critical subset -- Experiment F (the light, feasible decision-relevance proxy).

No simulator. The argument: on recorded data, the cases that would actually matter to a
planner -- pedestrians crossing into / near the ego path (PIE labels these) -- are exactly
where the models are worst. This connects the rarity/long-tail argument (Liu & Feng) to the
data without closing the loop.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def subset_mask(labels: dict, name: str, values: Iterable) -> np.ndarray:
    """Boolean mask of tracks whose label `name` is in `values`."""
    if name not in labels:
        raise KeyError(f"label {name!r} not in dump; available={list(labels)}")
    col = np.asarray(labels[name])
    values = set(np.asarray(list(values)).tolist())
    return np.array([v in values for v in col.tolist()])


def compare_subset(err: np.ndarray, mask: np.ndarray) -> dict:
    """Compare per-track error on the safety-critical subset vs the rest.

    Returns means and the ratio; >1 means the model is WORSE on the cases that matter.
    """
    err = np.asarray(err, float).ravel()
    mask = np.asarray(mask, bool).ravel()
    crit = err[mask]
    rest = err[~mask]
    out = {
        "n_critical": int(mask.sum()), "n_rest": int((~mask).sum()),
        "mean_critical": float(crit.mean()) if crit.size else float("nan"),
        "mean_rest": float(rest.mean()) if rest.size else float("nan"),
    }
    if rest.size and rest.mean() > 0:
        out["ratio_critical_over_rest"] = out["mean_critical"] / out["mean_rest"]
    else:
        out["ratio_critical_over_rest"] = float("nan")
    return out
