#!/bin/bash
nvidia-smi
echo "-------------------------"
export PYTHONUNBUFFERED=1

export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate vla0_blackwell

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

export HF_HOME=/vol/dissolve/matt/hf_cache
export HF_DATASETS_CACHE=/vol/dissolve/matt/hf_cache

cd /vol/dissolve/matt/models/vla0
PYTHON=/vol/dissolve/matt/envs/vla0_blackwell/bin/python
CUDA_VISIBLE_DEVICES=1 $PYTHON -m rv_train.train --exp-config configs/libero_object.yaml --resume 2>&1 | tee train_libero_object.log
