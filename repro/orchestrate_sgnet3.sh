#!/bin/bash
source /workspace/env.sh
export MPLBACKEND=Agg PYTHONPATH=/workspace/repos/tpb/SGNet:/workspace/repos/tpp/trajectron:/workspace/stubs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/repos/tpb/SGNet
echo "[sgnet3] waiting for Trajectron++ (TPP_DONE)..."; while [ ! -f /workspace/dumps/TPP_DONE ]; do sleep 30; done
echo "[sgnet3] start $(date)"
DUMPDIR=/workspace/dumps/ethucy; mkdir -p /workspace/dumps/logs_sgnet3
rm -f /workspace/dumps/FINAL_DONE
MAXJOBS=4; i=0
for split in eth hotel univ zara1 zara2; do
  gpu=$((i % 2)); SU=$(echo $split | tr a-z A-Z); lr=0.0001; [ "$split" = "eth" ] && lr=0.0005
  (
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    export TRAJCRIT_DUMP=$DUMPDIR/SGNetCVAE_${split}_seed1.npz
    export TRAJCRIT_MODEL=SGNetCVAE TRAJCRIT_DATASET=$split TRAJCRIT_SPLIT=test TRAJCRIT_SEED=1
    micromamba run -n tp python tools/ethucy/train_cvae.py --dataset $SU \
      --eth_root /workspace/data/ethucy_sgnet --ETH_CONFIG configs/ethucy/ETH_UCY.json \
      --epochs 50 --seed 1 --num_workers 4 --gpu $gpu --lr $lr \
      > /workspace/dumps/logs_sgnet3/${split}.log 2>&1
  ) &
  i=$((i+1)); while [ "$(jobs -r|wc -l)" -ge $MAXJOBS ]; do wait -n; done
done
wait; touch /workspace/dumps/FINAL_DONE; echo "[sgnet3] FINAL_DONE $(date)"
