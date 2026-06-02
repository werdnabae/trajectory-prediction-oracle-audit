"""Best-of-K oracle inflation -- Experiment B.

minADE_K reports the error of the BEST of K sampled trajectories, chosen using the
ground-truth future. That selection is impossible at deployment (the car does not know the
future). This module (1) shows minADE_K falling as K grows -- accuracy you can buy by
sampling more -- and (2) quantifies the gap between the oracle best-of-K and a deployable
selection (most-likely mode, or the expected error of a single random sample).

Requires RAW per-sample errors `ade_NK` of shape (N, K), i.e. re-generated dumps -- the
legacy scalar .pkl files already collapsed the K samples to their min and cannot be used
here. (The deterministic-vs-generative variant comparison is the legacy-data stand-in.)
"""
from __future__ import annotations

import numpy as np


def minade_curve(ade_NK: np.ndarray, Ks=None, n_repeats: int = 50, seed: int = 0) -> dict:
    """Mean-over-tracks of (min over a random size-K subset of samples), for each K.

    Samples are exchangeable draws, so for K < K_total we average over random subsets to
    get a stable estimate of "what you'd have seen with only K samples".
    """
    ade_NK = np.asarray(ade_NK, float)
    N, Ktot = ade_NK.shape
    if Ks is None:
        Ks = [k for k in (1, 2, 5, 10, 15, 20, 50) if k <= Ktot]
        if Ktot not in Ks:
            Ks.append(Ktot)
    rng = np.random.default_rng(seed)
    out = {}
    for K in Ks:
        if K >= Ktot:
            out[K] = float(ade_NK.min(axis=1).mean())
            continue
        vals = np.empty(n_repeats)
        for r in range(n_repeats):
            cols = rng.integers(0, Ktot, size=K)          # with-replacement subset
            vals[r] = ade_NK[:, cols].min(axis=1).mean()
        out[K] = float(vals.mean())
    return out


def selection_gap(ade_NK: np.ndarray, probs_NK: np.ndarray = None) -> dict:
    """Compare oracle best-of-K against deployable selection strategies (mean over tracks).

    oracle           : min over K (the published minADE_K number) -- NOT deployable
    expected_random  : mean over K (expected error of one random sample) -- deployable
    most_likely      : sample with highest predicted probability -- deployable (needs probs)
    inflation        : expected_random / oracle  (how much the oracle flatters the model)
    """
    ade_NK = np.asarray(ade_NK, float)
    oracle = float(ade_NK.min(axis=1).mean())
    expected_random = float(ade_NK.mean())   # mean over K then over N
    res = {"oracle_minK": oracle, "expected_random": expected_random,
           "inflation_x": expected_random / oracle if oracle > 0 else float("nan")}
    if probs_NK is not None:
        probs_NK = np.asarray(probs_NK, float)
        ml_idx = probs_NK.argmax(axis=1)
        res["most_likely"] = float(ade_NK[np.arange(ade_NK.shape[0]), ml_idx].mean())
    return res


def deterministic_vs_generative(det_err: np.ndarray, gen_err: np.ndarray) -> dict:
    """Legacy-data stand-in for Exp B: gap between a deterministic variant (single-shot, no
    oracle) and its generative sibling (best-of-K). Pure-numbers framing; the gap mixes the
    genuine modelling change with the oracle effect, so report it as an upper bound.
    """
    det = np.asarray(det_err, float).ravel()
    gen = np.asarray(gen_err, float).ravel()
    return {
        "det_mean": float(det.mean()), "gen_mean": float(gen.mean()),
        "ratio_det_over_gen": float(det.mean() / gen.mean()) if gen.mean() > 0 else float("nan"),
    }


def medoid_selection(pred, gt, chunk=2000):
    """Deployable 'sample-consensus' selector (a proxy for most-likely-mode without likelihoods).

    Per track, choose the sample closest in trajectory space to the other K-1 samples (the medoid,
    i.e. the densest mode) using ONLY the predictions, then return that sample's per-track ADE
    against ground truth. This is a stronger, still-deployable selector than a random/expected
    single sample, used to test robustness of the constant-velocity reversal.
    """
    from .schema import _centers
    pc = _centers(np.asarray(pred, float))   # (N,K,T,2)
    gc = _centers(np.asarray(gt, float))     # (N,T,2)
    N = pc.shape[0]
    out = np.empty(N)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        b = pc[s:e]                                          # (n,K,T,2)
        d = np.linalg.norm(b[:, :, None] - b[:, None], axis=-1).mean(-1)  # (n,K,K)
        med = d.sum(-1).argmin(1)                            # (n,) medoid sample per track
        sel = b[np.arange(e - s), med]                       # (n,T,2)
        out[s:e] = np.linalg.norm(sel - gc[s:e], axis=-1).mean(-1)
    return out
