tr='/workspace/repos/tpb/BiTraP/bitrap/engine/trainer.py'
s=open(tr).read()
if 'TRAJCRIT_DUMP' not in s:
    anchor="        all_gt_trajs = np.concatenate(all_gt_trajs, axis=0)\n"
    assert anchor in s, "trainer anchor not found"
    block=(
"        # === trajectory_eval raw dump (added) ===\n"
"        import os as _os, json as _json\n"
"        _dp = _os.environ.get('TRAJCRIT_DUMP')\n"
"        if _dp:\n"
"            _pred = np.transpose(np.asarray(all_pred_trajs), (0,2,1,3))\n"
"            _gt = np.asarray(all_gt_trajs)\n"
"            _meta = {k: _os.environ.get('TRAJCRIT_'+k.upper(), '') for k in ['model','dataset','split','seed']}\n"
"            np.savez_compressed(_dp, pred=_pred.astype('float32'), gt=_gt.astype('float32'),\n"
"                                track_id=np.arange(_pred.shape[0]), meta_json=np.array(_json.dumps(_meta)))\n"
"            print('TRAJCRIT_DUMP wrote', _dp, 'pred', _pred.shape, 'gt', _gt.shape)\n"
"            return\n"
"        # === end trajectory_eval dump ===\n")
    open(tr,'w').write(s.replace(anchor, anchor+block,1)); print('patched trainer.py')
else: print('trainer.py already patched')

tp='/workspace/repos/tpb/BiTraP/tools/train.py'
s=open(tp).read()
if 'TRAJCRIT_SEED' not in s:
    anchor="    cfg.merge_from_list(args.opts)\n"
    assert anchor in s, "train anchor not found"
    block=(
"    import random as _rnd\n"
"    _seed = int(os.environ.get('TRAJCRIT_SEED', '0'))\n"
"    _rnd.seed(_seed); np.random.seed(_seed); torch.manual_seed(_seed); torch.cuda.manual_seed_all(_seed)\n")
    open(tp,'w').write(s.replace(anchor, anchor+block,1)); print('patched train.py')
else: print('train.py already patched')
