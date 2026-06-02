#!/bin/bash
source /workspace/env.sh
export MPLBACKEND=Agg
cd /workspace/repos/tpp/trajectron
echo "[tpp] waiting for SGNet ALL_DONE..."; while [ ! -f /workspace/dumps/ALL_DONE ]; do sleep 30; done
echo "[tpp] start $(date)"
PROC=/workspace/repos/tpp/experiments/processed
MODELS=/workspace/repos/tpp/experiments/pedestrians/models
DUMPDIR=/workspace/dumps/ethucy
rm -f /workspace/dumps/TPP_DONE
i=0
for split in eth hotel univ zara1 zara2; do
  gpu=$((i % 2))
  (
    export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6
    tag=tpp_${split}
    micromamba run -n tp python train.py --conf $MODELS/${split}_vel/config.json \
      --data_dir $PROC --train_data_dict ${split}_train.pkl --eval_data_dict ${split}_test.pkl \
      --train_epochs 50 --vis_every 100000 --eval_every 100000 --save_every 50 \
      --batch_size 256 --preprocess_workers 6 --device cuda:$gpu --eval_device cuda:$gpu \
      --log_dir /workspace/dumps/tpp_logs --log_tag $tag --seed 1 \
      > /workspace/dumps/logs_tpp/${split}.log 2>&1
    MDIR=$(ls -d /workspace/dumps/tpp_logs/models_*${tag} | tail -1)
    micromamba run -n tp python /workspace/tpp_dump.py --model_dir "$MDIR" --ts 50 \
      --conf $MODELS/${split}_vel/config.json --data $PROC/${split}_test.pkl \
      --out $DUMPDIR/Trajectron++_${split}_seed1.npz --dataset $split --seed 1 \
      >> /workspace/dumps/logs_tpp/${split}.log 2>&1
  ) &
  i=$((i+1))
done
wait
touch /workspace/dumps/TPP_DONE
echo "[tpp] DONE $(date)"
