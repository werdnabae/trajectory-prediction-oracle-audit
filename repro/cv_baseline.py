import os, sys, json
sys.path.append('.'); sys.path.append('./datasets')
import numpy as np
from configs import cfg
from datasets import make_dataloader
split = sys.argv[1]
cfg.merge_from_file('configs/bitrap_np_ETH.yml')
cfg.merge_from_list(['DATASET.TRAJECTORY_PATH','/workspace/repos/tpp/experiments/processed',
                     'DATASET.ETH_CONFIG','configs/ETH_UCY.json','DATASET.NAME',split,
                     'DATALOADER.NUM_WORKERS','4','USE_WANDB','False'])
dl = make_dataloader(cfg,'test')
Xs=[]; Ys=[]
for batch in dl:
    Xs.append(batch['input_x'].numpy()); Ys.append(batch['target_y'].numpy())
X=np.concatenate(Xs,0); Y=np.concatenate(Ys,0)
pos=X[...,:2]; p=pos[:,-1]; v=pos[:,-1]-pos[:,-2]
T=Y.shape[1]; steps=np.arange(1,T+1)[None,:,None]
cv=p[:,None,:]+v[:,None,:]*steps
a=(pos[:,-1]-pos[:,-2])-(pos[:,-2]-pos[:,-3])
ca=p[:,None,:]+v[:,None,:]*steps+0.5*a[:,None,:]*steps**2
for name,pred in [('ConstVel',cv),('ConstAcc',ca)]:
    out=f'/workspace/dumps/ethucy/{name}_{split}.npz'
    meta={'model':name,'dataset':split,'split':'test','seed':'0'}
    np.savez_compressed(out, pred=pred[:,None].astype('float32'), gt=Y.astype('float32'),
                        track_id=np.arange(len(Y)), meta_json=np.array(json.dumps(meta)))
    dd=np.linalg.norm(pred[:,None]-Y[:,None],axis=-1)
    print(f'{name} {split}: N={len(Y)} ADE={dd.mean(-1).min(-1).mean():.3f} FDE={dd[...,-1].min(-1).mean():.3f}')
