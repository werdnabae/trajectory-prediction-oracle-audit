"""Qualitative figure: for a few representative tracks, show the K=20 predicted samples, the
ground truth, the oracle's pick (best-of-20), the deployable medoid sample, and constant velocity.
Visualizes why best-of-K looks good (a lucky sample hugs GT) while any deployable choice does not.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajectory_eval import io

MODEL, SCENE = "BiTrapNP", "univ"   # BiTraP shares the CV dump's dataloader -> tracks are index-aligned
MODEL_DISP, SCENE_DISP = "BiTraP", "Univ"   # display names matching the paper/tables
d = io.load_dump(f"dumps/ethucy/{MODEL}_{SCENE}_seed1.npz")
pred, gt = d.pred, d.gt                                  # (N,20,T,2), (N,T,2)
dd = np.linalg.norm(pred - gt[:, None], axis=-1).mean(-1)   # (N,20) per-sample ADE
oracle, oidx = dd.min(1), dd.argmin(1)

# constant velocity, index-aligned to the same tracks (verified below)
cv = io.load_dump(f"dumps/ethucy/ConstVel_{SCENE}.npz")
pcv, gcv = cv.pred[:, 0], cv.gt                          # (M,T,2),(M,T,2)
assert len(gcv) == len(gt), f"track-count mismatch {len(gcv)} vs {len(gt)}"
print("gt index-alignment max |diff|:", float(np.abs(gcv - gt).max()), "(want ~0)")


def medoid_idx(P):
    D = np.linalg.norm(P[:, None] - P[None], axis=-1).mean(-1)
    return D.sum(1).argmin()


def cv_for(i):
    return pcv[i]


order = np.argsort(oracle)
picks = [order[int(p * len(order))] for p in (0.55, 0.80, 0.92)]   # medium/hard/very-hard

# common half-span so all three panels render as equal-sized squares (equal aspect, no distortion,
# no stray vertical white space between panels of differing data range)
centers, half = [], 0.0
for i in picks:
    pts = np.concatenate([pred[i].reshape(-1, 2), gt[i], pcv[i]], 0)
    lo, hi = pts.min(0), pts.max(0)
    centers.append((lo + hi) / 2.0); half = max(half, (hi - lo).max() / 2.0 * 1.08)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
for ax, i, c in zip(axes, picks, centers):
    P, G = pred[i], gt[i]
    for k in range(P.shape[0]):
        ax.plot(P[k, :, 0], P[k, :, 1], color="0.72", lw=0.8, alpha=0.7, zorder=1,
                label="20 samples" if k == 0 else None)
    mi = medoid_idx(P); cvp = cv_for(i)
    med_ade = float(np.linalg.norm(P[mi] - G, axis=-1).mean())
    ax.plot(G[:, 0], G[:, 1], "k-", lw=2.6, label="ground truth", zorder=5)
    ax.plot(P[oidx[i], :, 0], P[oidx[i], :, 1], color="tab:green", lw=2.2,
            label="oracle pick (best-of-20)", zorder=4)
    ax.plot(P[mi, :, 0], P[mi, :, 1], color="tab:orange", lw=2.2, label="medoid (deployable)", zorder=4)
    ax.plot(cvp[:, 0], cvp[:, 1], color="tab:red", lw=2.0, ls="--", label="constant velocity", zorder=3)
    ax.scatter([G[0, 0]], [G[0, 1]], c="k", s=22, zorder=6)
    ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"oracle {oracle[i]:.2f} m  ·  medoid {med_ade:.2f} m  ·  CV {np.linalg.norm(cvp-G,axis=-1).mean():.2f} m",
                 fontsize=9)
axes[0].legend(fontsize=8, loc="best", framealpha=0.9)
fig.suptitle(f"{MODEL_DISP} on ETH/UCY ({SCENE_DISP}): the oracle keeps the one lucky sample; deployable selectors and constant velocity do not", fontsize=10)
fig.tight_layout()
os.makedirs("outputs", exist_ok=True)
fig.savefig("outputs/ethucy_qualitative.png", dpi=160)
print("wrote outputs/ethucy_qualitative.png ; picks(oracle ADE):", [round(float(oracle[i]), 2) for i in picks])
