import sys, os, json, argparse
sys.path.append('/workspace/repos/sgan')
import numpy as np, torch
class AttrDict(dict):
    __getattr__=dict.__getitem__
from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.utils import relative_to_abs, get_dset_path
ap=argparse.ArgumentParser(); ap.add_argument('--ckpt'); ap.add_argument('--split'); ap.add_argument('--outdir'); ap.add_argument('--K',type=int,default=20)
a=ap.parse_args()
ck=torch.load(a.ckpt, map_location='cuda'); args=AttrDict(ck['args'])
g=TrajectoryGenerator(obs_len=args.obs_len,pred_len=args.pred_len,embedding_dim=args.embedding_dim,
  encoder_h_dim=args.encoder_h_dim_g,decoder_h_dim=args.decoder_h_dim_g,mlp_dim=args.mlp_dim,num_layers=args.num_layers,
  noise_dim=args.noise_dim,noise_type=args.noise_type,noise_mix_type=args.noise_mix_type,pooling_type=args.pooling_type,
  pool_every_timestep=args.pool_every_timestep,dropout=args.dropout,bottleneck_dim=args.bottleneck_dim,
  neighborhood_size=args.neighborhood_size,grid_size=args.grid_size,batch_norm=args.batch_norm)
g.load_state_dict(ck['g_state']); g.cuda(); g.train()  # matches official SGAN eval sampling
dpath=get_dset_path(args.dataset_name,'test'); _,loader=data_loader(args,dpath)
P=[]; G=[]; CV=[]
with torch.no_grad():
  for batch in loader:
    batch=[t.cuda() for t in batch]
    obs,gt,obs_rel,gt_rel,nl,lm,sse=batch
    samples=[]
    for _ in range(a.K):
      pr=g(obs,obs_rel,sse); pa=relative_to_abs(pr,obs[-1]); samples.append(pa.cpu().numpy())
    s=np.transpose(np.stack(samples,0),(2,0,1,3))   # (n,K,T,2)
    gtn=gt.permute(1,0,2).cpu().numpy()             # (n,T,2)
    o=obs.permute(1,0,2).cpu().numpy()              # (n,obs,2)
    last=o[:,-1]; vel=o[:,-1]-o[:,-2]; T=gtn.shape[1]
    cv=last[:,None,:]+vel[:,None,:]*np.arange(1,T+1)[None,:,None]   # (n,T,2)
    P.append(s); G.append(gtn); CV.append(cv[:,None])
pred=np.concatenate(P,0); gt=np.concatenate(G,0); cv=np.concatenate(CV,0)
os.makedirs(a.outdir,exist_ok=True)
def save(name,arr,model):
    meta={'model':model,'dataset':a.split,'split':'test','seed':'0'}
    np.savez_compressed(os.path.join(a.outdir,name),pred=arr.astype('float32'),gt=gt.astype('float32'),
                        track_id=np.arange(len(gt)),meta_json=np.array(json.dumps(meta)))
save(f'SocialGAN_{a.split}_seed1.npz',pred,'SocialGAN')
save(f'ConstVelSG_{a.split}_seed1.npz',cv,'ConstVelSG')
dd=np.linalg.norm(pred-gt[:,None],axis=-1); ddc=np.linalg.norm(cv-gt[:,None],axis=-1)
print('SGAN',a.split,'N',len(gt),'minADE',round(float(dd.mean(-1).min(-1).mean()),3),'single',round(float(dd.mean(-1).mean(-1).mean()),3),'CV',round(float(ddc.mean(-1).min(-1).mean()),3))
