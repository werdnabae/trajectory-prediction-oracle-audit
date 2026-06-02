# Model variants: deterministic vs. generative (and why it matters for the paper)

You half-remembered a real and important distinction. Each model family ships a
**deterministic** variant and one or more **generative / multimodal** variants, and they are
evaluated *differently*. That difference is itself one of the paper's cleanest arguments.

## BiTraP (Yao et al., RA-L 2021)

| Variant | What it outputs | Multimodal? | How it's scored |
|---|---|---|---|
| **BiTraP-D** | a **single** trajectory (deterministic). Goal/endpoint estimated, then bi-directional decoding (forward from the last observation + backward from the goal). | No | **single-shot** — one prediction, one error. No oracle. |
| **BiTraP-NP** | a CVAE with a **categorical latent**; draw K=20 latents → 20 trajectories. "NP" = the future distribution is **non-parametric**, represented by the sample set. | Yes | **best-of-20** (minADE/minFDE): keep the single sample closest to ground truth. |
| **BiTraP-GMM** | a CVAE that emits the parameters of a per-timestep **Gaussian Mixture Model** — an explicit *parametric* predictive distribution. | Yes | best-of-20 for ADE/FDE; also supports likelihood (KDE-NLL). |

## SGNet (Wang et al., RA-L 2022)

| Variant | What it outputs | Multimodal? | How it's scored |
|---|---|---|---|
| **SGNet** | stepwise goal estimator → a **single** trajectory. | No | single-shot. |
| **SGNet-CVAE** | adds a CVAE on top of the stepwise goals → K=20 samples. | Yes | best-of-20. |

**The one axis that matters:** *deterministic (single-shot, honest)* vs. *generative
(K samples, scored by the oracle best-of-K)*. The deterministic variant has to commit to one
future and is graded on it; the generative variant gets 20 guesses and is graded on its
luckiest one.

## Why this belongs in the paper (it's Experiment B, for free)

Comparing a deterministic variant against its own generative sibling is a **within-model
natural experiment**: same authors, same backbone, same data, same training pipeline. Almost
the only thing that changes is *single-shot vs. best-of-20*. So the gap between them is a
near-clean readout of what the best-of-20 oracle buys.

From your **own committed PIE results** (computed by `trajectory_eval`, metric = bbox MSE @1.5s):

| Group | BiTraP-D (single-shot) | BiTraP-NP (best-of-20) | Ratio |
|---|---|---|---|
| adult | 493.4 | 91.9 | **5.4×** lower |
| child | 1593.9 | 417.6 | 3.8× lower |
| elderly | 683.6 | 172.7 | 4.0× lower |
| all | 511.6 | 99.7 | **5.1×** lower |

A model looks **~5× better** mostly by being allowed to draw 20 samples and keep the best one
against the (unknowable-at-deployment) ground truth.

**Be precise (so a reviewer can't push back):** this ratio is an *upper bound* on the oracle
effect, because BiTraP-NP also differs from BiTraP-D by being a genuinely different
generative model. To *decompose* "real modelling gain" vs. "oracle inflation," re-run NP
inference dumping all 20 samples and compute `minADE_K` for K = 1…20 (`trajectory_eval.bestofk`):
the K=1 / mean-over-K number is deployable; the K=20 min is the oracle; the gap between them
is the pure cherry-picking tax. That is the full Experiment B.

## Footnote you can use

When we recomputed BiTraP-NP's per-track errors from the saved run, the multimodal numbers
drifted from the published table (e.g. PIE child 418 vs. 457) while the *deterministic*
numbers matched to <1% (adult 493 vs. 490). That drift is the best-of-20 sampling noise
showing up directly — the generative scores are not even reproducible run-to-run, which is
its own small indictment of the protocol.
