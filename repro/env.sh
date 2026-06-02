export MAMBA_ROOT_PREFIX=/workspace/micromamba
export PATH=/workspace/bin:$PATH
export PIP_CACHE_DIR=/workspace/.cache/pip
export HF_HOME=/workspace/.cache/hf
export TORCH_HOME=/workspace/.cache/torch
export TMPDIR=/workspace/tmp
eval "$(/workspace/bin/micromamba shell hook -s bash)"
export PYTHONPATH=/workspace/stubs:$PYTHONPATH
