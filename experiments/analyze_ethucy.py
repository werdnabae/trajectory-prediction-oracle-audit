"""Experiments A-D on the ETH/UCY dumps, across all models present.

Reads dumps/ethucy/{MODEL}_{split}_seed{seed}.npz  (K samples, world meters), MODEL in
{BiTrapNP, SGNetCVAE}, plus ConstVel_{split}.npz / ConstAcc_{split}.npz (K=1).

For each (model, split) computes, averaged over seeds:
  best-of-K (oracle minADE)      -- the number papers report
  single-sample (deployable)     -- expected error of one sample (mean over K)
  oracle inflation = single/oracle
  seed std of the best-of-K number
and compares both to the constant-velocity baseline.

Outputs (outputs/):
  results_ethucy.json, ethucy_deploy_vs_oracle.png (the centerpiece), ethucy_bestofk.png
and prints the cross-architecture tables.
"""
from __future__ import annotations

import glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajectory_eval import io, tails, bestofk, stats  # noqa: E402

DUMPDIR = "dumps/ethucy"
OUT = "outputs"
SPLITS = ["eth", "hotel", "univ", "zara1", "zara2"]
os.makedirs(OUT, exist_ok=True)

# Display names (match the paper/tables) and a fixed panel order (Table 1 row order).
DISPLAY = {"SocialGAN": "Social GAN", "Trajectron++": "Trajectron++", "SGNetCVAE": "SGNet",
           "BiTrapNP": "BiTraP", "AgentFormer": "AgentFormer"}
SCENE_DISPLAY = {"eth": "ETH", "hotel": "Hotel", "univ": "Univ", "zara1": "Zara1", "zara2": "Zara2"}
PLOT_ORDER = ["SocialGAN", "Trajectron++", "SGNetCVAE", "BiTrapNP", "AgentFormer"]


def _ordered(models):
    return [m for m in PLOT_ORDER if m in models] + [m for m in models if m not in PLOT_ORDER]


def discover_models():
    models = set()
    for p in glob.glob(os.path.join(DUMPDIR, "*_seed*.npz")):
        m = re.match(r"(.+)_(?:eth|hotel|univ|zara1|zara2)_seed\d+\.npz", os.path.basename(p))
        if m and not m.group(1).startswith("Const"):   # exclude CV/CA baselines
            models.add(m.group(1))
    return sorted(models)


def cv_for_model(model, split):
    # Social GAN uses its own preprocessing -> compare to CV on its own tracks
    cands = [f"ConstVelSG_{split}_seed1.npz"] if model == "SocialGAN" else [f"ConstVel_{split}.npz"]
    for c in cands:
        p = os.path.join(DUMPDIR, c)
        if os.path.exists(p):
            return float(io.load_dump(p).metrics()["minade"].mean())
    return None


def main():
    models = discover_models()
    print("models found:", models)
    results = {}
    for model in models:
        results[model] = {}
        for split in SPLITS:
            paths = sorted(glob.glob(os.path.join(DUMPDIR, f"{model}_{split}_seed*.npz")))
            if not paths:
                continue
            b20, single, curves, track_best = [], [], [], None
            for p in paths:
                ade = io.load_dump(p).metrics()["ade"]      # (N, K)
                b20.append(float(ade.min(1).mean()))
                single.append(float(ade.mean()))
                curves.append(bestofk.minade_curve(ade))
                track_best = ade.min(1) if track_best is None else track_best
            ss = stats.seed_summary(b20)
            Ks = sorted(curves[0])
            results[model][split] = {
                "n_seeds": ss["n_seeds"], "best20_mean": ss["mean"], "best20_std": ss["std"],
                "single_mean": float(np.mean(single)),
                "oracle_inflation": float(np.mean(single) / ss["mean"]) if ss["mean"] else None,
                "cv": cv_for_model(model, split),
                "bestofk": {int(k): float(np.mean([c[k] for c in curves])) for k in Ks},
                "tail": tails.summarize(track_best),
            }

    # ---- tables ----
    for model in models:
        print(f"\n=== {model} (minADE, world meters) ===")
        print(f"{'split':7} {'best-of-K':18} {'single(deploy)':14} {'CV':7} {'oracle infl':11} {'CV<deploy?':10}")
        for split in SPLITS:
            r = results[model].get(split)
            if not r:
                continue
            cvwins = "YES" if (r["cv"] is not None and r["cv"] < r["single_mean"]) else "no"
            print(f"{split:7} {r['best20_mean']:.3f} ± {r['best20_std']:.3f}      "
                  f"{r['single_mean']:<14.3f} {r['cv'] or float('nan'):<7.3f} {r['oracle_inflation']:<11.2f} {cvwins}")

    print("\n=== KEY CLAIM: constant velocity vs each model's DEPLOYABLE single sample ===")
    for model in models:
        wins = sum(1 for s in SPLITS if (r := results[model].get(s)) and r["cv"] is not None and r["cv"] < r["single_mean"])
        tot = sum(1 for s in SPLITS if results[model].get(s))
        infl = np.mean([results[model][s]["oracle_inflation"] for s in SPLITS if results[model].get(s)])
        print(f"  {model}: CV beats deployable single sample on {wins}/{tot} splits; mean oracle inflation {infl:.2f}x")

    json.dump(results, open(os.path.join(OUT, "results_ethucy.json"), "w"), indent=2)
    _plots(results, models)
    print(f"\nWrote {OUT}/results_ethucy.json + figures")


def _plots(results, models):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
                         "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12})
    order = _ordered(models)

    def grid():
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), squeeze=False)
        flat = axes.ravel()
        for ax in flat[len(order):]:
            ax.axis("off")
        return fig, flat

    # centerpiece: per model, grouped bars (oracle / single / CV) across scenes; shared y so the
    # cross-model widening of the gap is read off honestly, one shared legend.
    fig, flat = grid()
    ymax = max(results[m][s]["single_mean"] for m in order for s in SPLITS if results[m].get(s)) * 1.08
    handles = labels = None
    for ax, model in zip(flat, order):
        splits = [s for s in SPLITS if results[model].get(s)]
        x = np.arange(len(splits)); w = 0.27
        oracle = [results[model][s]["best20_mean"] for s in splits]
        ostd = [results[model][s]["best20_std"] for s in splits]
        single = [results[model][s]["single_mean"] for s in splits]
        cv = [results[model][s]["cv"] for s in splits]
        ax.bar(x - w, oracle, w, yerr=ostd, capsize=3, label="best-of-K (oracle, reported)", color="#1f4e79")
        ax.bar(x, single, w, label="single sample (deployable)", color="#8aa9c9")
        ax.bar(x + w, cv, w, label="constant velocity", color="#c0504d")
        ax.set_xticks(x); ax.set_xticklabels([SCENE_DISPLAY[s] for s in splits])
        ax.set_title(DISPLAY.get(model, model)); ax.set_ylim(0, ymax); ax.grid(axis="y", alpha=0.3)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("minADE (m)")
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    flat[len(order)].legend(handles, labels, loc="center", frameon=True, fontsize=13)
    fig.suptitle("ETH/UCY: reported oracle number vs deployable reality vs constant velocity")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "ethucy_deploy_vs_oracle.png"), dpi=150); plt.close(fig)

    # best-of-K curves (one panel per model); per-scene CV reference drawn in the matching colour,
    # one shared scene legend.
    fig, flat = grid()
    handles = labels = None
    for ax, model in zip(flat, order):
        for s in SPLITS:
            r = results[model].get(s)
            if not r:
                continue
            ks = sorted(r["bestofk"])
            line, = ax.plot(ks, [r["bestofk"][k] for k in ks], marker="o", label=SCENE_DISPLAY[s])
            if r["cv"]:
                ax.axhline(r["cv"], ls=":", lw=1.3, alpha=0.6, color=line.get_color())
        ax.set_xlabel("K"); ax.set_title(DISPLAY.get(model, model)); ax.grid(alpha=0.3)
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("minADE$_K$ (m)")
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    flat[len(order)].legend(handles, labels, loc="center", frameon=True, fontsize=13, title="scene")
    fig.suptitle("best-of-K minADE vs number of samples K (dotted: per-scene constant-velocity reference)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "ethucy_bestofk.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
