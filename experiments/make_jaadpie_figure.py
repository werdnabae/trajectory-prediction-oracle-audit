"""Generate the JAAD/PIE deploy-vs-oracle figure (Fig.~\\ref{fig:jaadpie}) from the raw dumps.

Bar heights are recomputed from dumps/jaadpie/ via experiments/audit_jaadpie.py, so the figure always
matches Table~\\ref{tab:cross}. Run from the repo root:  python experiments/make_jaadpie_figure.py
Writes outputs/jaadpie_deploy.png.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from audit_jaadpie import audit, DUMPDIR, ROWS   # noqa: E402

C_or, C_si, C_me, C_cv, C_de = "#1f4e79", "#8fadc9", "#e8a33d", "#c0504d", "#4ea072"


def build_groups():
    groups = []
    for data, model, fn in ROWS:
        r = audit(os.path.join(DUMPDIR, fn))
        if r["K"] == 1:                                   # deterministic: committed + CV only
            bars = [(r["single"], C_de), (r["cv"], C_cv)]
        else:
            bars = [(r["best"], C_or), (r["single"], C_si), (r["medoid"], C_me), (r["cv"], C_cv)]
        groups.append((f"{model}\n{data}", bars))
    return groups


if __name__ == "__main__":
    groups = build_groups()
    fig, ax = plt.subplots(figsize=(11, 4.2)); w = 0.8 / 4
    for gi, (_, bars) in enumerate(groups):
        n = len(bars); offs = (np.arange(n) - (n - 1) / 2.0) * w
        for (val, col), off in zip(bars, offs):
            ax.bar(gi + off, val, w, color=col, edgecolor="white", linewidth=0.4)
    ax.axvline(2.5, color="0.7", linestyle=":", linewidth=1)        # JAAD | PIE divider
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("center ADE (px)")
    ax.set_title("JAAD / PIE: reported oracle vs deployable reality vs constant velocity")
    ax.grid(axis="y", color="0.9", linewidth=0.6); ax.set_axisbelow(True)
    leg = [Patch(facecolor=C_or, label="best-of-20 (oracle, reported)"),
           Patch(facecolor=C_si, label="single sample (deployable)"),
           Patch(facecolor=C_me, label="medoid (deployable)"),
           Patch(facecolor=C_de, label="committed (deterministic, BiTraP-D)"),
           Patch(facecolor=C_cv, label="constant velocity")]
    ax.legend(handles=leg, ncol=2, fontsize=9, framealpha=0.95, loc="upper left")
    fig.tight_layout()
    out = os.path.join(ROOT, "outputs", "jaadpie_deploy.png")
    fig.savefig(out, dpi=150); print("wrote", out)
