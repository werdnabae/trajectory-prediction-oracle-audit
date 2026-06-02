"""Per-scene breakdown table and a minADE_K diffuseness sweep (best-of-K vs single/medoid/CV)."""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajectory_eval import io, bestofk

DUMPDIR = "dumps/ethucy"
SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
MODELS = ["SocialGAN", "Trajectron++", "SGNetCVAE", "BiTrapNP", "AgentFormer"]
PRETTY = {"SocialGAN": "Social GAN", "Trajectron++": "Trajectron++", "SGNetCVAE": "SGNet",
          "BiTrapNP": "BiTraP", "AgentFormer": "AgentFormer"}


def dump_path(model, scene):
    c = sorted(glob.glob(os.path.join(DUMPDIR, f"{model}_{scene}_seed*.npz")))
    return c[0] if c else None


def cv_mean(model, scene):
    f = f"ConstVelSG_{scene}_seed1.npz" if model == "SocialGAN" else f"ConstVel_{scene}.npz"
    p = os.path.join(DUMPDIR, f)
    return float(io.load_dump(p).metrics()["minade"].mean()) if os.path.exists(p) else float("nan")


print("=== PER-SCENE (minADE, m): best20 / single / medoid / CV / rho ===")
for model in MODELS:
    for scene in SCENES:
        p = dump_path(model, scene)
        if not p:
            continue
        d = io.load_dump(p); ade = d.metrics()["ade"]
        b20 = float(ade.min(1).mean()); single = float(ade.mean())
        med = float(bestofk.medoid_selection(d.pred, d.gt).mean()); cv = cv_mean(model, scene)
        print(f"{PRETTY[model]:13} {scene:6} {b20:.3f} {single:.3f} {med:.3f} {cv:.3f} {single/b20:.2f}")

print("\n=== minADE_K sweep (pooled), K = 1,2,5,10,20 ===")
print(f"{'model':13} {'K=1':7}{'K=2':7}{'K=5':7}{'K=10':7}{'K=20':7}  drop(1->20)")
for model in MODELS:
    A = []
    for scene in SCENES:
        p = dump_path(model, scene)
        if p:
            A.append(io.load_dump(p).metrics()["ade"])
    A = np.concatenate(A, 0)
    c = bestofk.minade_curve(A, Ks=[1, 2, 5, 10, 20], n_repeats=50)
    print(f"{PRETTY[model]:13} " + "".join(f"{c[k]:.3f}  " for k in [1, 2, 5, 10, 20]) +
          f"  {c[1]/c[20]:.2f}x")
