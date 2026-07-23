#!/bin/bash
#SBATCH --job-name=vla0_realworld
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/vla0_realworld_relative_%j.out
#SBATCH --error=logs/vla0_realworld_relative_%j.err

cd /lus/lfs1aip2/projects/u6lm/mdp25/vla0

mkdir -p logs

source /home/u6lm/mdp25.u6lm/miniforge3/etc/profile.d/conda.sh
conda activate vla0

# Store all HuggingFace/LeRobot caches on Lustre (large quota) not home directory
export HF_LEROBOT_HOME=/lus/lfs1aip2/projects/u6lm/mdp25/vla0/data/lerobot
export HF_HOME=/lus/lfs1aip2/projects/u6lm/mdp25/vla0/data/hf_home
export HF_DATASETS_CACHE=/lus/lfs1aip2/projects/u6lm/mdp25/vla0/data/hf_home/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Log GPU memory every 2 seconds to diagnose OOM
# nvidia-smi dmon -s mu -d 2 > logs/gpu_mem_${SLURM_JOB_ID}.log 2>&1 &
# NVMON_PID=$!

# Resume from last checkpoint if one exists
RESUME_FLAG=""
if [ -f ./runs/realworld_reproduce_relative/model_last.pth ]; then
    RESUME_FLAG="--resume"
    echo "Found existing checkpoint, resuming training"
fi

python -u -m rv_train.train \
    --exp-config ./configs/realworld_reproduce_relative.yaml \
    --devices 0,1,2,3 \
    $RESUME_FLAG

# kill $NVMON_PID 2>/dev/null || true