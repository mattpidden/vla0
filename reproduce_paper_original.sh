#!/bin/bash
nvidia-smi
echo "-------------------------"
export PYTHONUNBUFFERED=1

export PATH="$HOME/miniconda3/bin:$PATH"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate vla0_blackwell

# torchcodec needs FFmpeg/CUDA libs from the conda env on the linker path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/lib:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH

python -c "import sys; print(sys.executable); print(sys.path)"

export HF_HOME=/vol/dissolve/matt/hf_cache
export HF_DATASETS_CACHE=/vol/dissolve/matt/hf_cache
hf auth whoami

cd /vol/dissolve/matt/models/vla0
PYTHON=/vol/dissolve/matt/envs/vla0_blackwell/bin/python
CUDA_VISIBLE_DEVICES=1 $PYTHON -m rv_train.train --exp-config configs/paper_original.yaml --resume 2>&1 | tee -a reproduce_paper_original.log
