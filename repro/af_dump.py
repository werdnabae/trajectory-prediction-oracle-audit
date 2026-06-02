import sys, os, json, argparse
sys.path.append('/workspace/repos/agentformer')
os.chdir('/workspace/repos/agentformer')
import numpy as np, torch
from data.dataloader import data_generator
from utils.config import Config
from model.model_lib import model_dict
from utils.utils import prepare_seed
ap=argparse.ArgumentParser(); ap.add_argument('--cfg'); ap.add_argument('--split_name'); ap.add_argument('--out'); ap.add_argument('--gpu',type=int,default=0)
a=ap.parse_args()
cfg=Config(a.cfg)
device=torch.device('cuda',a.gpu); torch.cuda.set_device(a.gpu); torch.set_grad_enabled(False)
prepare_seed(cfg.seed)
log=open('/dev/null','w')
model=model_dict[cfg.get('model_id','agentformer')](cfg); model.set_device(device); model.eval()
ep=cfg.get_last_epoch(); cp=torch.load(cfg.model_path%ep, map_location='cpu'); model.load_state_dict(cp['model_dict'], strict=False)
print('loaded epoch', ep, 'sample_k', cfg.sample_k, 'traj_scale', cfg.traj_scale)
gen=data_generator(cfg, log, split='test', phase='testing'); K=cfg.sample_k
P=[]; G=[]
while not gen.is_epoch_end():
    data=gen()
    if data is None: continue
    gt=torch.stack(data['fut_motion_3D'],dim=0).to(device)*cfg.traj_scale     # (n,T,2)
    model.set_data(data)
    sample,_=model.inference(mode='infer', sample_num=K, need_weights=False)
    sample=sample.transpose(0,1).contiguous()*cfg.traj_scale                  # (K,n,T,2)
    pm=data['pred_mask']
    for i in range(gt.shape[0]):
        if pm is not None and pm[i]!=1.0: continue
        P.append(sample[:,i].cpu().numpy()); G.append(gt[i].cpu().numpy())
pred=np.stack(P,0); gt=np.stack(G,0)
meta={'model':'AgentFormer','dataset':a.split_name,'split':'test','seed':'0'}
np.savez_compressed(a.out, pred=pred.astype('float32'), gt=gt.astype('float32'), track_id=np.arange(len(gt)), meta_json=np.array(json.dumps(meta)))
dd=np.linalg.norm(pred-gt[:,None],axis=-1)
print('AF',a.split_name,'N',len(gt),'pred',pred.shape,'minADE',round(float(dd.mean(-1).min(-1).mean()),3),'single',round(float(dd.mean(-1).mean(-1).mean()),3))
