#!/bin/bash
source /workspace/env.sh
export MPLBACKEND=Agg
cd /workspace/repos/tpb/BiTraP
SPLITS="eth hotel univ zara1 zara2"
SEEDS="1 2 3 4 5"
MAXJOBS=10
DUMPDIR=/workspace/dumps/ethucy; mkdir -p $DUMPDIR /workspace/dumps/logs /workspace/dumps/ckpt /workspace/dumps/out
rm -f /workspace/dumps/DONE
i=0
for split in $SPLITS; do for seed in $SEEDS; do
  gpu=$((i % 2))
  (
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
    export TRAJCRIT_DUMP=$DUMPDIR/BiTrapNP_${split}_seed${seed}.npz
    export TRAJCRIT_MODEL=BiTraPNP TRAJCRIT_DATASET=$split TRAJCRIT_SPLIT=test TRAJCRIT_SEED=$seed
    micromamba run -n tp python tools/train.py --gpu $gpu --config_file configs/bitrap_np_ETH.yml \
      DATASET.TRAJECTORY_PATH /workspace/repos/tpp/experiments/processed DATASET.ETH_CONFIG configs/ETH_UCY.json \
      DATASET.NAME $split SOLVER.MAX_EPOCH 50 DATALOADER.NUM_WORKERS 4 USE_WANDB False MODEL.K 20 \
      CKPT_DIR /workspace/dumps/ckpt/${split}_s${seed} OUT_DIR /workspace/dumps/out/${split}_s${seed} \
      > /workspace/dumps/logs/${split}_s${seed}.log 2>&1
  ) &
  i=$((i+1))
  while [ "$(jobs -r | wc -l)" -ge $MAXJOBS ]; do wait -n; done
done; done
wait
touch /workspace/dumps/DONE
echo "ALL DONE $(date)"
