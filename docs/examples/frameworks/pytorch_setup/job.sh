#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00

set -e  # exit on error.
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

# First, in general, it's a good idea to run `uv sync` once before submitting
# jobs to ensure that the venv is created and all the dependencies are installed.
# On cluster with internet access on compute nodes (Mila cluster), this saves a bit
# of GPU compute time at the start of your job since dependencies already be in the cache.
# On DRAC or PAICE clusters where you don't have internet access on compute nodes,
# if you need packages that are not in the DRAC wheelhouse, you should run `uv sync`
# on a login node once before submitting the job, and use `uv run --offline python main.py`

# On a login / interactive node:
# ```
# srun --pty --gpus=1 --mem=16G --time=00:10:00 uv sync
# ```

uv run python main.py
