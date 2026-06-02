# GPU campaign: run matrix + machine specs

The expensive step is only **training × seeds** (and a few inference re-dumps). Every model
run must dump a standardized `.npz` (see `scripts/INTEGRATION.md`) holding **raw per-track
predictions**; all analysis (Experiments A–G) then runs offline on those files via `trajectory_eval`.

## What we can do with ZERO new compute (already verified)

Your repo commits the trained checkpoints (`checkpoints/{JAAD,PIE,TITAN}/*.pth`) **and** the
per-track error `.pkl` files. So these run today on a laptop:
- **Exp A** (significance / CIs / effect sizes) and **Exp C** (heavy tail) on BiTraP-D,
  BiTraP-NP, SGNet × JAAD/PIE/TITAN — `experiments/demo_existing_pkls.py`.
- **Exp B legacy stand-in** (deterministic vs best-of-20) — see `docs/model_variants.md`.

## What needs INFERENCE only (minutes of GPU, uses existing checkpoints)

The committed `.pkl`s store only scalar MSE, not raw trajectories. To get raw 20-sample
predictions for the *full* Experiments B/D/E on JAAD/PIE, re-run **inference** with the
existing checkpoints, patched to dump `.npz`:
- **Exp B (full)**: BiTraP-NP, SGNet-CVAE — dump all K=20 samples → `minADE_K` sweep.
- **Exp D**: add the Constant-Velocity / Constant-Acceleration baselines (no training).
- **Exp E**: dump predicted vs. GT boxes → pixel-error-vs-bbox-size confound.
- **Exp F**: attach PIE crossing labels to the dump → safety-critical subset.
Needs the **JAAD + PIE annotation files** (free download). TITAN needs a re-request to Honda.

## What needs TRAINING (the real campaign, with seeds for Exp A)

Train each model **5 seeds** per dataset/split, dump `.npz` each time.

| Regime | Datasets | Models | Runs | Notes |
|---|---|---|---|---|
| BEV meters (pedestrian) | ETH/UCY (5 leave-one-out splits) | CV, CA, Social GAN, Trajectron++, BiTraP-NP, SGNet-CVAE, **AgentFormer** | ~5 models × 5 splits × 5 seeds ≈ **125** (+CV/CA free) | tiny + fast (minutes each) |
| Pixel bbox | JAAD, PIE | BiTraP-D/NP, SGNet/-CVAE, +CV | ~30 | checkpoints exist; mostly re-dumps + a few retrains for seeds |
| BEV + maps (AV) | nuScenes | Trajectron++, SGNet, **AgentFormer** | ~3 models × 3 seeds ≈ **9** | heavier; shows critiques hold in the real-AV regime |

Total ≈ **300–400 GPU-hours**, dominated by ETH/UCY's many short runs.

## Time-boxed "~2 hour" session (recommended first rental)

The wall-clock bottleneck is **setup + data download + nuScenes preprocessing**, NOT GPU
compute (the models are tiny). To fit ~2 hours, **defer nuScenes and AgentFormer** (each adds
hours of download/preprocess/new-env) and do the two things that need almost no time:

1. **JAAD/PIE inference re-dumps** with the existing checkpoints (Exp B full / D / E / F).
   Annotations are small; inference is minutes. ~30–40 min including download.
2. **ETH/UCY seed-variance training** for the Exp A headline forest plot. Runs are minutes
   each; **run 4 concurrently** (each uses <4 GB on a 24 GB card) and start with **3 seeds**
   (extend to 5 later). ~30–45 min for the whole grid.

Budget: ~20–30 min setup (prebuilt PyTorch image) + the above ≈ **under 2 hours**. nuScenes +
AgentFormer become an optional later session for the "AV-regime generality" claim.

## Recommended vast.ai machine

| Resource | Spec | Why |
|---|---|---|
| **GPU** | 1× **RTX 3090 or 4090 (24 GB)** | these models are small (your original ran on a 3080 10 GB); 24 GB gives headroom for AgentFormer / nuScenes batch sizes. A 16 GB card (A4000 / 4060 Ti 16 GB) also works if budget-tight. **One GPU is enough** — parallelism is across cheap runs, not within. |
| **vCPU** | 8–16 | dataloading + nuScenes/`trajdata` preprocessing is CPU-bound |
| **RAM** | 32 GB min, **64 GB** recommended | nuScenes / trajdata caching is memory-hungry |
| **Disk** | **150–250 GB** | ETH/UCY <1 GB; JAAD/PIE annotations ~tens of GB (you do **not** need raw video for trajectory prediction — bbox/feature files suffice); nuScenes prediction needs maps+meta (or the preprocessed Trajectron++ cache ~5–15 GB), **not** the full ~300 GB camera blobs |
| **CUDA image** | CUDA 11.x or 12.x base | rebuild envs (below) |

**Cost:** 3090 ≈ \$0.20–0.35/hr, 4090 ≈ \$0.35–0.70/hr on vast.ai → whole campaign ≈
**\$80–200**. Inference-only re-dumps are negligible.

**Software (important):** the original `environment.yaml` is Python 3.6 / cudatoolkit 11.0 /
old PyTorch. Do **not** force one env for everything. Build **separate conda envs per repo**
(BiTraP+SGNet can share the legacy env; Trajectron++ and AgentFormer each have their own).
`trajectory_eval` only needs numpy/scipy/matplotlib/pandas and reads the `.npz` dumps, so it is
env-independent — keep it in its own lightweight env for analysis.

## Suggested order (fastest signal first)

1. **Laptop, now:** A + C + B-stand-in on committed data (done — figures in `outputs/`).
2. **1–2 GPU-hours:** patch JAAD/PIE inference to dump `.npz`; run full B/D/E/F.
3. **Half a day:** ETH/UCY training × seeds → the headline **seed-variance forest plot** for A.
4. **Optional:** nuScenes + AgentFormer for the AV-regime generality claim.
