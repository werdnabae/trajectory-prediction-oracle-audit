"""JAAD/PIE ego-view bounding-box audit (Table~\\ref{tab:cross}).

Per model: best-of-20 (oracle), expected single sample, medoid, constant velocity, oracle inflation
rho, and the worst-5% tail share. Boxes are stored cxcywh, so the deployable center is (cx, cy)
directly (the shared ``trajectory_eval`` ``_centers`` assumes x1y1x2y2, so this driver takes the center
itself). Reads the raw K-sample dumps in dumps/jaadpie/ (regenerable via repro/patch_jaadpie.py).

Run from the repo root:  python experiments/audit_jaadpie.py
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajectory_eval import tails

DUMPDIR = "dumps/jaadpie"
ROWS = [("JAAD", "BiTraP-NP",  "rawdump_BiTraPNP_JAAD.npz"),
        ("JAAD", "SGNet-CVAE", "rawdump_SGNetCVAE_JAAD.npz"),
        ("JAAD", "BiTraP-D",   "rawdump_BiTraPD_JAAD.npz"),
        ("PIE",  "BiTraP-NP",  "rawdump_BiTraPNP_PIE.npz"),
        ("PIE",  "SGNet-CVAE", "rawdump_SGNetCVAE_PIE.npz"),
        ("PIE",  "BiTraP-D",   "rawdump_BiTraPD_PIE.npz")]


def centers_pred(pred):                     # (N,T,K,4) cxcywh -> (N,K,T,2)
    return np.transpose(pred, (0, 2, 1, 3))[..., :2]


def medoid_ade(pc, gc):                     # pc (N,K,T,2), gc (N,T,2)
    N = pc.shape[0]; out = np.empty(N)
    for s in range(0, N, 2000):
        e = min(s + 2000, N); b = pc[s:e]
        d = np.linalg.norm(b[:, :, None] - b[:, None], axis=-1).mean(-1)   # (n,K,K)
        mk = d.sum(-1).argmin(1)
        out[s:e] = np.linalg.norm(b[np.arange(e - s), mk] - gc[s:e], axis=-1).mean(-1)
    return out


def audit(path):
    z = np.load(path)
    pred, gt, obs = (np.asarray(z[k], float) for k in ("pred", "gt", "obs"))
    pc, gc = centers_pred(pred), gt[..., :2]
    ade = np.linalg.norm(pc - gc[:, None], axis=-1).mean(2)                # (N,K)
    best, single = ade.min(1), ade.mean(1)
    med = medoid_ade(pc, gc)
    oc = obs[..., :2]; vel = oc[:, -1] - oc[:, -2]; T = gt.shape[1]
    cv = oc[:, -1][:, None, :] + np.arange(1, T + 1)[None, :, None] * vel[:, None, :]
    cv_ade = np.linalg.norm(cv - gc, axis=-1).mean(1)
    N = len(best); rng = np.random.default_rng(0); reps = np.empty(5000)
    for i in range(5000):
        idx = rng.integers(0, N, N); reps[i] = single[idx].mean() / best[idx].mean()
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return dict(K=ade.shape[1], best=best.mean(), single=single.mean(), medoid=med.mean(),
                cv=cv_ade.mean(), rho=single.mean() / best.mean(), lo=lo, hi=hi,
                tail=tails.tail_share(best, 0.05) * 100)


if __name__ == "__main__":
    hdr = ("Data", "Model", "best-20", "single", "medoid", "CV", "rho", "tail%")
    print(f"{hdr[0]:5}{hdr[1]:12}{hdr[2]:>8}{hdr[3]:>8}{hdr[4]:>8}{hdr[5]:>7}{hdr[6]:>7}{hdr[7]:>7}")
    for data, model, fn in ROWS:
        p = os.path.join(DUMPDIR, fn)
        if not os.path.exists(p):
            print(f"  MISSING {fn}"); continue
        r = audit(p)
        if r["K"] == 1:   # deterministic: committed prediction only (best=single=medoid)
            print(f"{data:5}{model:12}{'--':>8}{r['single']:>8.1f}{'--':>8}{r['cv']:>7.1f}{'--':>7}{'--':>7}")
        else:
            print(f"{data:5}{model:12}{r['best']:>8.1f}{r['single']:>8.1f}{r['medoid']:>8.1f}"
                  f"{r['cv']:>7.1f}{r['rho']:>7.2f}{r['tail']:>7.1f}  (rho 95% CI {r['lo']:.2f},{r['hi']:.2f})")
