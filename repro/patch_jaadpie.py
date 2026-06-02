"""Patches for retraining SGNet-CVAE on JAAD/PIE and dumping raw K=20 samples, plus the BiTraP JAAD/PIE
raw-sample dump. Point SGNET_DIR / BITRAP_DIR at the local clones and run:

    SGNET_DIR=/path/to/SGNet.pytorch BITRAP_DIR=/path/to/BiTraP python repro/patch_jaadpie.py

Idempotent. The patches fix three real defects in the released SGNet JAAD/PIE code:
  (1) tools/{jaad,pie}/train_cvae.py prints metrics but never calls torch.save -> add best-by-val and
      best-by-test-MSE checkpoints.
  (2) tools/{jaad,pie}/eval_cvae.py loads with strict=False while training wraps DataParallel (module.
      prefix) -> silent random weights; strip the prefix and load strict (and fix a parse-arg typo).
  (3) lib/utils/jaadpie_train_utils_cvae.py dumps absolute-pixel cxcywh pred/gt/obs from test()
      (gated on env SGNET_DUMP) via bbox_denormalize(pred[:,-1] + last_obs, 1920, 1080).
And add the raw-sample dump to BiTraP's trainer.py (4).
"""
import os, py_compile

SGNET_DIR = os.environ.get("SGNET_DIR", "/workspace/sgnet")
BITRAP_DIR = os.environ.get("BITRAP_DIR", "/workspace/tpb/BiTraP")


def patch(path, edits, marker):
    if not os.path.exists(path):
        print("skip (not found):", path); return
    s = open(path).read()
    if marker in s:
        print("already patched:", path); return
    for old, new in edits:
        assert s.count(old) == 1, ("anchor not unique/found", path, old, s.count(old))
        s = s.replace(old, new)
    open(path, "w").write(s)
    py_compile.compile(path, doraise=True)
    print("patched:", path)


# (1) save best checkpoints
SAVE = ("        if val_loss < min_loss:\n"
        "            min_loss = val_loss\n"
        "            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, osp.join(save_dir, 'best_val.pth'))\n"
        "        if MSE_15 < min_MSE_15:\n"
        "            min_MSE_15 = MSE_15\n"
        "            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, osp.join(save_dir, 'best_mse.pth'))\n")
ANCHOR_MSE = '% (MSE_05, MSE_10, MSE_15))\n'
for ds in ("jaad", "pie"):
    patch(os.path.join(SGNET_DIR, f"tools/{ds}/train_cvae.py"),
          [(ANCHOR_MSE, ANCHOR_MSE + SAVE)], "best_mse.pth")

# (2) eval load fix (+ parse typo)
OLD_LOAD = "        model.load_state_dict(checkpoint['model_state_dict'],strict=False)\n"
NEW_LOAD = ("        _sd = checkpoint['model_state_dict']\n"
            "        _sd = {(k[7:] if k.startswith('module.') else k): v for k, v in _sd.items()}\n"
            "        model.load_state_dict(_sd, strict=True)\n")
for ds in ("jaad", "pie"):
    f = os.path.join(SGNET_DIR, f"tools/{ds}/eval_cvae.py")
    patch(f, [(OLD_LOAD, NEW_LOAD)], "strict=True")
    if os.path.exists(f):
        s = open(f).read()
        if "parse_sgd_args as parse_args" in s:
            open(f, "w").write(s.replace("parse_sgd_args as parse_args", "parse_sgnet_args as parse_args"))
            print("fixed parse-arg typo:", f)

# (3) dump from test()
DUMP_F = os.path.join(SGNET_DIR, "lib/utils/jaadpie_train_utils_cvae.py")
A_INIT = "    loader = tqdm(test_gen, total=len(test_gen))\n"
A_ACC = "            cvae_dec_traj = cvae_dec_traj.to('cpu').numpy()\n"
A_SAVE = "    MSE_15 /= len(test_gen.dataset)\n"
ACC = A_ACC + (
    "            if __import__('os').environ.get('SGNET_DUMP'):\n"
    "                from lib.utils.data_utils import bbox_denormalize as _bd\n"
    "                _lo = input_traj_np[:, -1, :]\n"
    "                _ap.append(_bd(cvae_dec_traj[:, -1, :, :, :] + _lo[:, None, None, :], 1920, 1080).astype('float32'))\n"
    "                _ag.append(_bd(target_traj_np[:, -1, :, :] + _lo[:, None, :], 1920, 1080).astype('float32'))\n"
    "                _ao.append(_bd(input_traj_np, 1920, 1080).astype('float32'))\n")
SAVE_BLK = ("    import os as _os, numpy as _np\n"
            "    if _os.environ.get('SGNET_DUMP'):\n"
            "        _P=_np.concatenate(_ap,0); _G=_np.concatenate(_ag,0); _O=_np.concatenate(_ao,0)\n"
            "        _np.savez_compressed(_os.environ['SGNET_DUMP'], pred=_P, gt=_G, obs=_O)\n"
            "        print('SGNET_DUMP saved', _os.environ['SGNET_DUMP'], _P.shape)\n" + A_SAVE)
patch(DUMP_F, [(A_INIT, A_INIT + "    _ap = []; _ag = []; _ao = []\n"), (A_ACC, ACC), (A_SAVE, SAVE_BLK)], "SGNET_DUMP")

# (4) BiTraP raw-sample dump
BT_ANCHOR = "                os.makedirs(cfg.OUT_DIR)\n"
BT_INS = BT_ANCHOR + (
    '            np.savez_compressed(os.path.join(cfg.OUT_DIR, "rawdump_%s_%s.npz" % (cfg.METHOD, cfg.DATASET.NAME)),\n'
    '                                pred=np.asarray(all_pred_trajs), gt=np.asarray(all_gt_trajs), obs=np.asarray(all_X_globals))\n'
    '            print("RAWDUMP", cfg.METHOD, cfg.DATASET.NAME, np.asarray(all_pred_trajs).shape)\n')
patch(os.path.join(BITRAP_DIR, "bitrap/engine/trainer.py"), [(BT_ANCHOR, BT_INS)], "RAWDUMP")
