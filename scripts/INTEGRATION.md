# Wiring `trajectory_eval` dumps into each model repo

Goal: each model's eval loop writes ONE standardized `.npz` per run, holding **raw per-track
predictions** — not its own ADE/FDE. `trajectory_eval` then computes every metric identically across
models. This is what makes the numbers comparable (the opposite of the paper's
incomparable-protocol critique).

## The contract

At the end of an eval pass, collect for the whole test split:
- `pred`  — `(N, K, T, D)` predicted futures (K=1 if deterministic). For multimodal models,
  keep **all K samples** (do NOT pre-reduce to best-of-K — that throws away Experiment B).
- `gt`    — `(N, T, D)` ground-truth futures.
- `D = 2` for world-meter xy (ETH/UCY, nuScenes); `D = 4` for pixel bbox `(x1,y1,x2,y2)`
  (JAAD/PIE/TITAN).
- optional `labels` dict of `(N,)` arrays: `age`, `gender`, `crossing`, … (for Exp F).

Then:

```python
from trajectory_eval.schema import DumpMeta, save_dump
save_dump(
    f"dumps/{model}_{dataset}_{split}_seed{seed}.npz",
    pred=pred, gt=gt,
    meta=DumpMeta(model=model, dataset=dataset, split=split, seed=seed,
                  coord_space="pixel_bbox" if dataset in ("JAAD","PIE","TITAN") else "world_m",
                  obs_len=obs_len, pred_len=pred_len),
    labels={"age": age_arr, "crossing": crossing_arr},  # whatever you have
)
```

## BiTraP (`BiTraP/bitrap/engine/`)

The inference engine already accumulates per-track MSE (that's what made the legacy `.pkl`s).
Find the loop that iterates batches and computes errors; it already has the predicted and
ground-truth tensors in hand (BiTraP works in `(x,y,w,h)` pixels for JAAD/PIE). Accumulate
the raw arrays into lists and `np.concatenate` them, then `save_dump(...)`. For BiTraP-NP keep
the full sample axis (K=20). Convert `(x,y,w,h)` → `(x1,y1,x2,y2)` once, consistently, before
dumping (or dump `(x,y,w,h)` and tell `compute_metrics` nothing changes — center is the same;
just be consistent).

## SGNet (`SGNet/tools/{jaad,pie,ethucy}/`)

The test scripts compute MSE on `(x1,y1,x2,y2)` corners (JAAD/PIE) or world xy (ETH/UCY).
Same recipe: stash the raw `pred`/`gt` batches, concatenate, `save_dump`. SGNet-CVAE: keep K.

## Constant-Velocity / -Acceleration baselines (Exp D)

No repo needed. From each dataset's observed history `X (N, obs, D)`:
- CV: last step + last velocity × horizon. CA: add ½·accel·t².
Build `pred (N,1,T,D)`, reuse the same `gt`, `save_dump(model="ConstVel", ...)`. Run this
inside each dataset's dataloader so the tracks line up with the deep models.

## Trajectron++ / AgentFormer (ETH/UCY, nuScenes)

Both already output world-meter trajectories and K samples. Patch their evaluation entry
point to concatenate raw `pred (N,K,T,2)` and `gt (N,T,2)` and `save_dump(coord_space="world_m")`.

## Sanity check after dumping

```python
from trajectory_eval import io
d = io.load_dump("dumps/BiTrapNP_PIE_test_seed0.npz")
print(d.pred.shape, d.gt.shape, d.meta)
print({k: float(v.mean()) for k, v in d.metrics().items() if v.ndim == 1})
```
Compare the deterministic model's `minade`/`corner_mse` against the legacy `.pkl` mean — they
should match to within rounding. If they don't, the dump is wrong, not the toolkit.
