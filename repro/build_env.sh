#!/bin/bash
set -e
source /workspace/env.sh
echo "[build] creating env tp (py3.9)"
micromamba create -y -n tp python=3.9 >/dev/null 2>&1
echo "[build] installing torch 2.2.2 cu121"
micromamba run -n tp pip install --no-input torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
echo "[build] installing support deps (numpy<2 for old code)"
micromamba run -n tp pip install --no-input "numpy==1.23.5" "scipy==1.10.1" pyyaml yacs dill termcolor tqdm pandas scikit-learn future six pillow opencv-python-headless
echo "[build] verifying torch + GPU"
micromamba run -n tp python -c "import torch; print('torch', torch.__version__, 'cuda_ok', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo "[build] DONE"
