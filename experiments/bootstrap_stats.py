"""Bootstrap confidence intervals and a stronger (medoid) deployable selector.

For each model: pooled-track 95% bootstrap CIs on best-of-20 and on the deployable single sample,
the oracle-inflation ratio with a paired-bootstrap CI, the sample-consensus (medoid) deployable
error, and how often constant velocity beats the expected-single and the medoid selectors.
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajectory_eval import io, stats, bestofk, tails

DUMPDIR = "dumps/ethucy"
SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
MODELS = ["SocialGAN", "Trajectron++", "SGNetCVAE", "BiTrapNP", "AgentFormer"]


def cv_tracks(model, scene):
    f = f"ConstVelSG_{scene}_seed1.npz" if model == "SocialGAN" else f"ConstVel_{scene}.npz"
    p = os.path.join(DUMPDIR, f)
    return io.load_dump(p).metrics()["minade"] if os.path.exists(p) else None


hdr = ("model", "best20[95%CI]", "single[95%CI]", "medoid", "CV", "rho[95%CI]",
       "CV<single", "CV<medoid", "tail%")
print(f"{hdr[0]:13}{hdr[1]:20}{hdr[2]:20}{hdr[3]:8}{hdr[4]:7}{hdr[5]:18}{hdr[6]:10}{hdr[7]:10}{hdr[8]}")
for model in MODELS:
    b20a, sina, meda, cva = [], [], [], []
    cw_s = cw_m = nsc = 0
    for scene in SCENES:
        cands = sorted(glob.glob(os.path.join(DUMPDIR, f"{model}_{scene}_seed*.npz")))
        if not cands:
            continue
        d = io.load_dump(cands[0]); ade = d.metrics()["ade"]   # (N,K)
        b20 = ade.min(1); single = ade.mean(1)
        med = bestofk.medoid_selection(d.pred, d.gt)
        cv = cv_tracks(model, scene)
        b20a.append(b20); sina.append(single); meda.append(med)
        if cv is not None:
            cva.append(cv); nsc += 1
            cw_s += int(cv.mean() < single.mean())
            cw_m += int(cv.mean() < med.mean())
    B, S, M = np.concatenate(b20a), np.concatenate(sina), np.concatenate(meda)
    CVp = np.concatenate(cva)
    bci = stats.bootstrap_ci(B, n_boot=5000)
    sci = stats.bootstrap_ci(S, n_boot=5000)
    rng = np.random.default_rng(0); n = len(B); reps = np.empty(3000)
    for i in range(3000):
        idx = rng.integers(0, n, n); reps[i] = S[idx].mean() / B[idx].mean()
    rho = S.mean() / B.mean(); rlo, rhi = np.percentile(reps, [2.5, 97.5])
    tail = tails.tail_share(B, 0.05) * 100
    print(f"{model:13}{bci[0]:.3f}[{bci[1]:.3f},{bci[2]:.3f}]  "
          f"{sci[0]:.3f}[{sci[1]:.3f},{sci[2]:.3f}]  {M.mean():.3f}   {CVp.mean():.3f}  "
          f"{rho:.2f}[{rlo:.2f},{rhi:.2f}]  {cw_s}/{nsc:<8}{cw_m}/{nsc:<8}{tail:.1f}")
