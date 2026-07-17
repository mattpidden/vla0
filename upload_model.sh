#!/bin/bash
#SBATCH --job-name=hf_upload
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/lus/lfs1aip2/projects/u6lm/mdp25/vla0/logs/hf_upload_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/u6lm/mdp25/vla0/logs/hf_upload_%j.err

source /home/u6lm/mdp25.u6lm/miniforge3/etc/profile.d/conda.sh
conda activate vla0

hf upload mattpidden/vla0-realworld-epoch28 \
    /lus/lfs1aip2/projects/u6lm/mdp25/vla0/runs/realworld_reproduce/model_28/ \
    . \
    --repo-type model

echo "Upload exit code: $?"
