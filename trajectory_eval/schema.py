"""Standardized per-track dump format + uniform metric computation.

A dump stores RAW per-track predictions and ground truth so that *this module* -- not the
model repo -- computes ADE/FDE/MSE identically for every model. Metadata records the
coordinate space ('world_m' vs 'pixel_bbox') so world-meter and pixel results are never
silently mixed.

Array conventions
-----------------
pred : float array (N, K, T, D)   K prediction samples per track (K=1 if deterministic)
gt   : float array (N, T, D)      ground-truth future
D == 2 -> (x, y) position
D == 4 -> (x1, y1, x2, y2) bounding-box corners; ADE/FDE use the box CENTER,
          and an extra corner-MSE (matching the SenSys MSE) is also returned.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

VALID_COORD = ("world_m", "pixel_bbox", "pixel_xy")


@dataclass
class DumpMeta:
    model: str
    dataset: str
    split: str = "test"
    seed: int = 0
    group: str = "all"          # demographic / subset tag, "all" = whole test set
    coord_space: str = "world_m"
    obs_len: int = 0
    pred_len: int = 0
    protocol: str = "default"   # used by Experiment G
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.coord_space not in VALID_COORD:
            raise ValueError(f"coord_space must be one of {VALID_COORD}, got {self.coord_space!r}")


def _as_4d(pred: np.ndarray) -> np.ndarray:
    """Coerce pred to (N, K, T, D). Accept (N, T, D) deterministic -> K=1."""
    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim == 3:
        pred = pred[:, None, :, :]
    if pred.ndim != 4:
        raise ValueError(f"pred must be (N,K,T,D) or (N,T,D); got shape {pred.shape}")
    return pred


def _centers(a: np.ndarray) -> np.ndarray:
    """Map last-dim coordinates to a 2-D point. (...,2)->itself; (...,4)->box center."""
    D = a.shape[-1]
    if D == 2:
        return a
    if D == 4:
        x = (a[..., 0] + a[..., 2]) * 0.5
        y = (a[..., 1] + a[..., 3]) * 0.5
        return np.stack([x, y], axis=-1)
    raise ValueError(f"last dim must be 2 or 4, got {D}")


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Per-track displacement metrics, computed uniformly for every model.

    Returns a dict of numpy arrays (all length N unless noted):
      ade        (N, K)  per-sample average displacement error (L2 over time, then mean)
      fde        (N, K)  per-sample final displacement error
      minade     (N,)    min over K samples of ade        (the "best-of-K" headline number)
      minfde     (N,)    min over K samples of fde
      meanade    (N,)    mean over K of ade               (a deployable, non-oracle summary)
      corner_mse (N,)    only when D==4: mean squared error over box corners & time
    """
    pred = _as_4d(pred)                       # (N,K,T,D)
    gt = np.asarray(gt, dtype=np.float64)     # (N,T,D)
    if gt.ndim != 3:
        raise ValueError(f"gt must be (N,T,D); got {gt.shape}")
    if pred.shape[0] != gt.shape[0] or pred.shape[2:] != gt.shape[1:]:
        raise ValueError(f"pred {pred.shape} incompatible with gt {gt.shape}")

    pc = _centers(pred)                        # (N,K,T,2)
    gc = _centers(gt)[:, None, :, :]           # (N,1,T,2)
    d = np.linalg.norm(pc - gc, axis=-1)       # (N,K,T) L2 per timestep
    ade = d.mean(axis=2)                        # (N,K)
    fde = d[:, :, -1]                           # (N,K)

    out = {
        "ade": ade,
        "fde": fde,
        "minade": ade.min(axis=1),
        "minfde": fde.min(axis=1),
        "meanade": ade.mean(axis=1),
    }
    if pred.shape[-1] == 4:
        # corner MSE averaged over the 4 coords and time, best sample (min) by default
        se = (pred - gt[:, None, :, :]) ** 2    # (N,K,T,4)
        corner_mse_k = se.mean(axis=(2, 3))     # (N,K)
        out["corner_mse"] = corner_mse_k.min(axis=1)
        out["corner_mse_mean"] = corner_mse_k.mean(axis=1)
    return out


def save_dump(path: str, pred: np.ndarray, gt: np.ndarray, meta: DumpMeta,
              track_id: Optional[np.ndarray] = None, labels: Optional[dict] = None) -> None:
    """Write a standardized .npz dump. `labels` is a dict of (N,) arrays (age, gender, crossing...)."""
    pred = _as_4d(pred)
    gt = np.asarray(gt, dtype=np.float64)
    N = gt.shape[0]
    if track_id is None:
        track_id = np.arange(N)
    arrays = dict(pred=pred.astype(np.float32), gt=gt.astype(np.float32),
                  track_id=np.asarray(track_id))
    labels = labels or {}
    for k, v in labels.items():
        v = np.asarray(v)
        if len(v) != N:
            raise ValueError(f"label {k!r} has length {len(v)} != N={N}")
        arrays[f"label__{k}"] = v
    arrays["meta_json"] = np.array(json.dumps(asdict(meta)))
    np.savez_compressed(path, **arrays)
