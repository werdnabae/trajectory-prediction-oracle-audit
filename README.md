# Trajectory-prediction evaluation

Code to evaluate multimodal trajectory predictors by **what a vehicle would actually receive**, a
single committed prediction, rather than by best-of-$K$ displacement error, which keeps whichever
of $K$ sampled trajectories is closest to the (unknown-at-runtime) ground truth.

Given the raw $K$ samples dumped from any model, the library recomputes, in one place: the
deployable single-sample error, a sample-consensus (medoid) selector, the oracle-inflation ratio,
the best-of-$K$ curve, heavy-tail statistics (CVaR, worst-$k\%$ share, Gini), bootstrap confidence
intervals and paired tests, and a constant-velocity baseline. Recomputing every metric from raw
samples, instead of inheriting each repo's own evaluation code, makes results comparable across
models by construction.

## Layout
```
trajectory_eval/   library: schema, io, stats, tails, bestofk, safety, plots
experiments/       analysis scripts (analyze_ethucy.py, bootstrap_stats.py, per_scene_detail.py,
                   qualitative_figure.py, a legacy-PIE demo)
repro/             scripts + REPRO.md to regenerate per-model sample dumps on a GPU box
outputs/           example results (results_ethucy.json + figures)
scripts/, docs/    integration notes and methodology docs
```

## Setup
```
uv sync
```

## Run
```
uv run python experiments/analyze_ethucy.py    # over dumps/ethucy/*.npz -> outputs/
uv run python experiments/bootstrap_stats.py   # bootstrap CIs + medoid-selector comparison
uv run python experiments/per_scene_detail.py  # per-scene breakdown + minADE_K sweep
```

## Dump format
Standardized `.npz` (`trajectory_eval/schema.py`): `pred (N,K,T,D)`, `gt (N,T,D)`, `track_id`,
optional per-track `labels`, and `meta_json`. `D=2` for world-meter xy; `D=4` for pixel boxes.
Producing dumps from the models themselves is described in `repro/REPRO.md`.

## Example findings
On ETH/UCY, across Social GAN, Trajectron++, SGNet, BiTraP, and AgentFormer (world coordinates):
- best-of-20 is **2.5–4.8×** lower than a deployable single prediction;
- a **constant-velocity** baseline beats every model's deployable single sample on **all 25
  model–scene cells** (and a stronger sample-consensus selector on 20/25);
- per-track error is **heavy-tailed**: the worst 5% of tracks own ~20% of total error.
