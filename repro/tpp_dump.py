import sys, os, json, argparse, dill
sys.path.append('/workspace/repos/tpp/trajectron')
import numpy as np, torch
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
from utils.trajectory_utils import prediction_output_to_trajectories

ap=argparse.ArgumentParser()
ap.add_argument('--model_dir'); ap.add_argument('--ts',type=int); ap.add_argument('--conf')
ap.add_argument('--data'); ap.add_argument('--out'); ap.add_argument('--dataset'); ap.add_argument('--seed',default='0')
a=ap.parse_args()
dev='cuda:0'
env=dill.load(open(a.data,'rb'),encoding='latin1')
mr=ModelRegistrar(a.model_dir,dev); mr.load_models(a.ts)
hp=json.load(open(a.conf))
stg=Trajectron(mr,hp,None,dev); stg.set_environment(env); stg.set_annealing_params()
for attr in hp.get('override_attention_radius',[]):
    n1,n2,r=attr.split(' '); env.attention_radius[(n1,n2)]=float(r)
ph=hp['prediction_horizon']; max_h=hp['maximum_history_length']
for scene in env.scenes:
    scene.calculate_scene_graph(env.attention_radius, hp['edge_addition_filter'], hp['edge_removal_filter'])
preds=[]; gts=[]
with torch.no_grad():
    for scene in env.scenes:
        ts=np.arange(scene.timesteps)
        out=stg.predict(scene, ts, ph, num_samples=20, min_history_timesteps=7,
                        min_future_timesteps=12, z_mode=False, gmm_mode=False, full_dist=False)
        if not out: continue
        pred_dict,_,fut_dict=prediction_output_to_trajectories(out, scene.dt, max_h, ph, prune_ph_to_future=True)
        for t in pred_dict:
            for node in pred_dict[t]:
                p=np.asarray(pred_dict[t][node])
                if p.ndim==4: p=p[0]                    # (1,K,T,2)->(K,T,2)
                f=np.asarray(fut_dict[t][node])
                if f.shape[0]!=ph or p.shape[1]!=ph: continue
                preds.append(p); gts.append(f)
pred=np.stack(preds,0); gt=np.stack(gts,0)               # (N,K,T,2),(N,T,2)
meta={'model':'Trajectron++','dataset':a.dataset,'split':'test','seed':a.seed}
np.savez_compressed(a.out, pred=pred.astype('float32'), gt=gt.astype('float32'),
                    track_id=np.arange(len(gt)), meta_json=np.array(json.dumps(meta)))
dd=np.linalg.norm(pred-gt[:,None],axis=-1)
print('WROTE',a.out,'pred',pred.shape,'gt',gt.shape,'minADE',round(float(dd.mean(-1).min(-1).mean()),3),'minFDE',round(float(dd[...,-1].min(-1).mean()),3))
