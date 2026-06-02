"""Appendix audits: final-displacement (FDE) oracle inflation and tail concentration, every model.

Reproduces appendix Table 4 (ETH/UCY FDE), Table 5 (JAAD/PIE FDE) and Table 7 (tail concentration).
ETH/UCY reads the standardized dumps in dumps/ethucy/ via trajectory_eval; JAAD/PIE reads the cxcywh
bbox dumps in dumps/jaadpie/. The per-scene ETH/UCY ADE breakdown (Table 6) is produced separately by
experiments/per_scene_detail.py.

Run from the repo root:  python experiments/audit_appendix.py
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajectory_eval import io, tails
from trajectory_eval.schema import _centers

ETHDIR, JPDIR = "dumps/ethucy", "dumps/jaadpie"
SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
ETH_MODELS = [("SocialGAN", "Social GAN"), ("Trajectron++", "Trajectron++"), ("SGNetCVAE", "SGNet"),
              ("BiTrapNP", "BiTraP"), ("AgentFormer", "AgentFormer")]
JP_ROWS = [("JAAD", "BiTraP-NP", "rawdump_BiTraPNP_JAAD.npz"), ("JAAD", "SGNet-CVAE", "rawdump_SGNetCVAE_JAAD.npz"),
           ("JAAD", "BiTraP-D", "rawdump_BiTraPD_JAAD.npz"), ("PIE", "BiTraP-NP", "rawdump_BiTraPNP_PIE.npz"),
           ("PIE", "SGNet-CVAE", "rawdump_SGNetCVAE_PIE.npz"), ("PIE", "BiTraP-D", "rawdump_BiTraPD_PIE.npz")]


def medoid_idx(pc):                                  # pc (N,K,T,2) -> (N,) medoid sample index
    N = pc.shape[0]; out = np.empty(N, int)
    for s in range(0, N, 2000):
        e = min(s + 2000, N); b = pc[s:e]
        d = np.linalg.norm(b[:, :, None] - b[:, None], axis=-1).mean(-1)
        out[s:e] = d.sum(-1).argmin(1)
    return out


def eth_arrays(model):                               # per-track FDE best/single/medoid, CV-FDE, best-ADE
    fb, fs, fm, cv, ba = [], [], [], [], []
    for sc in SCENES:
        c = sorted(glob.glob(os.path.join(ETHDIR, f"{model}_{sc}_seed*.npz")))
        if not c: continue
        d = io.load_dump(c[0]); m = d.metrics(); fde = m["fde"]
        pc = _centers(np.asarray(d.pred, float)); gc = _centers(np.asarray(d.gt, float))
        mk = medoid_idx(pc)
        fb.append(fde.min(1)); fs.append(fde.mean(1)); ba.append(m["ade"].min(1))
        fm.append(np.linalg.norm(pc[np.arange(len(mk)), mk][:, -1] - gc[:, -1], axis=-1))
        cf = f"ConstVelSG_{sc}_seed1.npz" if model == "SocialGAN" else f"ConstVel_{sc}.npz"
        cp = os.path.join(ETHDIR, cf)
        if os.path.exists(cp): cv.append(io.load_dump(cp).metrics()["fde"].min(1))
    return tuple(np.concatenate(x) for x in (fb, fs, fm, cv, ba))


def jp_arrays(path):                                 # cxcywh bbox -> FDE best/single/medoid, CV-FDE, best-ADE
    z = np.load(path); pred, gt, obs = (np.asarray(z[k], float) for k in ("pred", "gt", "obs"))
    pc = np.transpose(pred, (0, 2, 1, 3))[..., :2]; gc = gt[..., :2]
    fde = np.linalg.norm(pc[:, :, -1, :] - gc[:, None, -1, :], axis=-1)
    ade = np.linalg.norm(pc - gc[:, None], axis=-1).mean(2)
    mk = medoid_idx(pc); fm = fde[np.arange(len(mk)), mk]
    oc = obs[..., :2]; vel = oc[:, -1] - oc[:, -2]; T = gt.shape[1]
    cvf = np.linalg.norm((oc[:, -1] + T * vel) - gc[:, -1], axis=-1)
    return fde.min(1), fde.mean(1), fm, cvf, ade.min(1)


def tailrow(name, best):
    print(f"  {name:18}{tails.tail_share(best,0.05)*100:8.1f}{tails.tail_share(best,0.01)*100:8.1f}{tails.gini(best):8.2f}")


if __name__ == "__main__":
    eth_bestade, jp_bestade = {}, {}
    print("== Table 4: ETH/UCY FDE audit (meters) ==")
    print(f"  {'Model':13}{'best-20':>8}{'single':>8}{'medoid':>8}{'CV':>7}{'rhoFDE':>8}")
    for key, disp in ETH_MODELS:
        fb, fs, fm, cv, ba = eth_arrays(key); eth_bestade[disp] = ba
        print(f"  {disp:13}{fb.mean():8.2f}{fs.mean():8.2f}{fm.mean():8.2f}{cv.mean():7.2f}{fs.mean()/fb.mean():8.2f}")

    print("== Table 5: JAAD/PIE FDE audit (pixels) ==")
    print(f"  {'Data':5}{'Model':12}{'best-20':>8}{'single':>8}{'medoid':>8}{'CV':>7}{'rhoFDE':>8}")
    for data, model, fn in JP_ROWS:
        p = os.path.join(JPDIR, fn)
        if not os.path.exists(p):
            print(f"  MISSING {fn}"); continue
        fb, fs, fm, cvf, ba = jp_arrays(p); jp_bestade[f"{data} {model}"] = ba
        if model == "BiTraP-D":   # deterministic: committed only
            print(f"  {data:5}{model:12}{'--':>8}{fs.mean():8.1f}{'--':>8}{cvf.mean():7.1f}{'--':>8}")
        else:
            print(f"  {data:5}{model:12}{fb.mean():8.1f}{fs.mean():8.1f}{fm.mean():8.1f}{cvf.mean():7.1f}{fs.mean()/fb.mean():8.2f}")

    print("== Table 7: tail concentration (worst-5% / worst-1% / Gini, best-of-20 error) ==")
    for disp in [d for _, d in ETH_MODELS]:
        tailrow("ETH/UCY " + disp, eth_bestade[disp])
    for name, ba in jp_bestade.items():
        tailrow(name, ba)
