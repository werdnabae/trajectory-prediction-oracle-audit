"""Run Experiments A (significance) and C (heavy-tail) on the EXISTING SenSys .pkl files,
plus the deterministic-vs-generative (best-of-K stand-in) comparison -- no GPU, no datasets.

Usage:
    python experiments/demo_existing_pkls.py --pkl-dir /path/to/Trajectory-Prediction-Bias/notebooks \
        --dataset PIE --metric MSE_15 --out outputs/

Every number here is computed by trajectory_eval, identically across models.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajectory_eval import io, stats, tails, bestofk, plots  # noqa: E402

MODELS = ["BiTrapD", "BiTrapNP", "SGNet"]
PRETTY = {"BiTrapD": "BiTraP-D (deterministic)",
          "BiTrapNP": "BiTraP-NP (best-of-20)",
          "SGNet": "SGNet (deterministic)"}


def hr(c="="):
    print(c * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl-dir", required=True, help="dir containing the SenSys *_test.pkl files")
    ap.add_argument("--dataset", default="PIE")
    ap.add_argument("--metric", default="MSE_15")
    ap.add_argument("--group", default="all")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # ---- load per-track errors for each model -------------------------------------------
    data = {}
    for m in MODELS:
        path = io.find_legacy(args.pkl_dir, m, args.dataset, args.group)
        if path is None:
            print(f"[skip] no file for {m} {args.dataset} {args.group}")
            continue
        data[m] = io.legacy_metric(path, args.metric)
    if len(data) < 2:
        print("Need >=2 models; aborting.")
        return

    hr()
    print(f"EXPERIMENT A -- statistical rigor   ({args.dataset}, metric={args.metric}, group={args.group})")
    hr()
    ci = {}
    for m, x in data.items():
        point, lo, hi = stats.bootstrap_ci(x, n_boot=args.n_boot)
        ci[m] = (point, lo, hi)
        print(f"  {PRETTY[m]:<28}  n={x.size:>6}  mean={point:>10.1f}   95% CI [{lo:.1f}, {hi:.1f}]")

    print("\n  Pairwise comparison (unpaired; positive diff => first model worse):")
    pairs, pv = [], []
    keys = list(data)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            r = stats.compare_unpaired(data[a], data[b], n_boot=args.n_boot)
            pairs.append((a, b, r))
            pv.append(r["mwu_p"])
    adj = stats.holm_bonferroni(pv)
    for (a, b, r), pa in zip(pairs, adj):
        print(f"    {a:>9} vs {b:<9}  diff={r['diff']:>10.1f}  CI[{r['diff_ci'][0]:.1f},{r['diff_ci'][1]:.1f}]"
              f"  MWU p={r['mwu_p']:.2e} (Holm {pa:.2e})  Cliff's d={r['cliffs_delta']:+.3f}")

    hr()
    print("EXPERIMENT C -- heavy tail")
    hr()
    summ = {}
    for m, x in data.items():
        s = tails.summarize(x)
        summ[m] = s
        print(f"  {PRETTY[m]}")
        print(f"     mean={s['mean']:.1f}  median={s['median']:.1f}  mean/median={s['mean_over_median']:.2f}"
              f"  P99={s['p99']:.1f}  max={s['max']:.1f}")
        print(f"     CVaR95={s['cvar95']:.1f}  worst-5%-own={s['tail_share_top5pct']*100:.1f}% of total error"
              f"  Gini={s['gini']:.3f}  excess-kurtosis={s['excess_kurtosis']:.1f}")

    rf = tails.rank_flip(summ, "mean", "cvar95")
    print("\n  Ranking by mean  :", " < ".join(PRETTY[m] for m in rf["order_by_mean"]))
    print("  Ranking by CVaR95:", " < ".join(PRETTY[m] for m in rf["order_by_cvar95"]))
    print(f"  -> ranking changed: {rf['ranking_changed']}  (Kendall tau={rf['kendall_tau']:.3f})")

    # ---- best-of-K stand-in: deterministic vs generative sibling ------------------------
    if "BiTrapD" in data and "BiTrapNP" in data:
        hr()
        print("EXPERIMENT B (legacy stand-in) -- deterministic vs best-of-20, per group")
        hr()
        for g in ["adult", "child", "elderly", "all"]:
            pd_ = io.find_legacy(args.pkl_dir, "BiTrapD", args.dataset, g)
            pn_ = io.find_legacy(args.pkl_dir, "BiTrapNP", args.dataset, g)
            if not pd_ or not pn_:
                continue
            r = bestofk.deterministic_vs_generative(
                io.legacy_metric(pd_, args.metric), io.legacy_metric(pn_, args.metric))
            print(f"  {g:<8} BiTraP-D={r['det_mean']:>9.1f}  BiTraP-NP={r['gen_mean']:>9.1f}"
                  f"  ->  {r['ratio_det_over_gen']:.1f}x lower with best-of-20")
        print("  (upper bound on the best-of-20 effect; full decomposition needs raw 20-sample dumps)")

    # ---- figures ------------------------------------------------------------------------
    plots.forest({PRETTY[m]: ci[m] for m in data}, xlabel=f"{args.metric} (pixel^2)",
                 title=f"{args.dataset}: mean {args.metric} with 95% bootstrap CI",
                 savepath=os.path.join(args.out, f"forest_{args.dataset}_{args.metric}.png"))
    plots.lorenz({PRETTY[m]: data[m] for m in data},
                 savepath=os.path.join(args.out, f"lorenz_{args.dataset}_{args.metric}.png"),
                 title=f"{args.dataset}: error concentration ({args.metric})")
    print(f"\nFigures written to {args.out}/")


if __name__ == "__main__":
    main()
