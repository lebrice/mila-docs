#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# NOTE: Run this either with `sbatch make_env.sh` or within an interactive job with `salloc`:
# salloc --gres=gpu:1 --cpus-per-task=1 --mem=16G --time=00:30:00

# Exit on error
set -e

module --quiet purge
module load anaconda/3
module load cuda/11.7

ENV_NAME=pytorch

## Create the environment (see the example):
conda create --yes --name $ENV_NAME python=3.9 pytorch torchvision torchaudio pytorch-cuda=11.7 --channel pytorch --channel nvidia
# Install as many packages as possible with Conda:
conda install --yes --name $ENV_NAME tqdm --channel conda-forge
# Activate the environment:
conda activate $ENV_NAME
# Install the rest of the packages with pip:
pip install rich
conda env export --no-builds --from-history --file environment.yaml