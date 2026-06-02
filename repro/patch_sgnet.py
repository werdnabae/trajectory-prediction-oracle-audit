f='/workspace/repos/tpb/SGNet/lib/utils/ethucy_train_utils_cvae.py'
s=open(f).read()
if 'TRAJCRIT_DUMP' in s: print('already patched'); raise SystemExit
if 'import os' not in s.split('\n')[0:15].__str__(): s='import os\n'+s
# init lists in test()
s=s.replace("    count = 0\n    model.eval()",
            "    count = 0\n    _dump=os.environ.get('TRAJCRIT_DUMP'); _P=[]; _G=[]\n    model.eval()",1)
# accumulate inside loop
anchor="            target_traj_np = target_traj.to('cpu').numpy()\n"
s=s.replace(anchor, anchor+"            if _dump: _P.append(cvae_dec_traj[:,-1]); _G.append(target_traj_np[:,-1])\n",1)
# dump after loop
anchor2="    ADE_08 /= count\n"
block=("    if _dump:\n"
"        import numpy as _np, json as _json\n"
"        _Pa=_np.transpose(_np.concatenate(_P,0),(0,2,1,3)); _Ga=_np.concatenate(_G,0)\n"
"        _meta={k:os.environ.get('TRAJCRIT_'+k.upper(),'') for k in ['model','dataset','split','seed']}\n"
"        _np.savez_compressed(_dump, pred=_Pa.astype('float32'), gt=_Ga.astype('float32'), track_id=_np.arange(len(_Ga)), meta_json=_np.array(_json.dumps(_meta)))\n"
"        print('TRAJCRIT_DUMP wrote', _dump, _Pa.shape, _Ga.shape)\n")
s=s.replace(anchor2, block+anchor2,1)
open(f,'w').write(s); print('patched ethucy_train_utils_cvae.py')
