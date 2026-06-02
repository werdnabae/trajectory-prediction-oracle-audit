"""Loaders: the new standardized .npz dumps AND the legacy SenSys .pkl files.

Legacy SenSys .pkl format (from the original repo's test.py): a dict of per-track arrays
  {'MSE_05','MSE_10','MSE_15','CMSE','CFMSE'}  -- pixel^2 bounding-box errors, one per track.
These only carry scalar per-track error (no raw trajectories), so they support the
significance (Exp A) and heavy-tail (Exp C) analyses but not best-of-K / constant-velocity,
which need re-generated raw dumps.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import schema

LEGACY_KEYS = ("MSE_05", "MSE_10", "MSE_15", "CMSE", "CFMSE")


def load_legacy_pkl(path: str) -> dict:
    """Load one SenSys result .pkl -> dict of 1-D float arrays."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected a dict, got {type(d)}")
    out = {}
    for k, v in d.items():
        out[k] = np.asarray(v, dtype=np.float64)
    return out


def legacy_metric(path: str, key: str = "MSE_15") -> np.ndarray:
    """Per-track values for one metric key from a legacy .pkl."""
    d = load_legacy_pkl(path)
    if key not in d:
        raise KeyError(f"{path}: metric {key!r} not found; available={list(d)}")
    return d[key]


def find_legacy(search_dirs, model: str, dataset: str, group: str = "all") -> Optional[str]:
    """Locate `{model}_{dataset}_{group}_test.pkl` under any of search_dirs (recursive)."""
    if isinstance(search_dirs, str):
        search_dirs = [search_dirs]
    name = f"{model}_{dataset}_{group}_test.pkl"
    for d in search_dirs:
        hits = glob.glob(os.path.join(d, "**", name), recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


@dataclass
class Dump:
    """A loaded standardized .npz dump."""
    pred: np.ndarray            # (N,K,T,D)
    gt: np.ndarray              # (N,T,D)
    track_id: np.ndarray        # (N,)
    meta: dict
    labels: dict                # name -> (N,) array

    def metrics(self) -> dict:
        return schema.compute_metrics(self.pred, self.gt)


def load_dump(path: str) -> Dump:
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    labels = {k[len("label__"):]: z[k] for k in z.files if k.startswith("label__")}
    return Dump(pred=z["pred"], gt=z["gt"], track_id=z["track_id"], meta=meta, labels=labels)
