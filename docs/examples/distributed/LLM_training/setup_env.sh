#!/bin/bash
set -o errexit

# Install miniconda if 'conda' is not available
function install_miniconda {
	wget "https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh" -O $HOME/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh
	echo "32d73e1bc33fda089d7cd9ef4c1be542616bd8e437d1f77afeeaf7afdb019787  $HOME/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh" | sha256sum -c -
	bash $HOME/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -u -p "$CONDA_INSTALL_PREFIX"
}

CONDA_INSTALL_PREFIX="$HOME/miniconda3"
export PATH="$PATH:$CONDA_INSTALL_PREFIX/condabin"

which conda >/dev/null || module load anaconda/3 || install_miniconda
module load cuda/11.7

(conda activate base 2>/dev/null) || eval "$(conda shell.bash hook)"

# NOTE: Use a temporary directory if you want to re-create the environment from scratch each time.
# CONDA_ENV_PREFIX=$SLURM_TMPDIR/env
CONDA_ENV_PREFIX=$HOME/conda/llm_training


if [ ! -d $CONDA_ENV_PREFIX ]; then
	# Create a conda environment and use the libmamba solver:
	conda create --solver=classic -y -p $CONDA_ENV_PREFIX python=3.9 conda conda-libmamba-solver -c conda-forge
	conda activate $CONDA_ENV_PREFIX
	while read f
	do
		if [[ -e "$f" ]]
		then
			export CONDA_EXE="$f"
			break
		fi
	done < <(hash -r; which -a conda)
	conda config --set solver libmamba

	# Install pytorch:
	conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 transformers datasets evaluate accelerate rich simple-parsing wandb -c pytorch -c nvidia
	# Install other conda packages:
	# conda install -y rich -c conda-forge
	# Install other pip packages:
	pip install "deepspeed>=0.8.2"
else
	conda activate $CONDA_ENV_PREFIX
fi

# Download dataset
export HF_DATASETS_CACHE=$SCRATCH/cache/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=$SCRATCH/cache/huggingface/hub
MODEL_NAME=${MODEL_NAME:="facebook/opt-2.7b"}
python3 -c "import datasets ; datasets.load_dataset('wikitext', 'wikitext-103-v1')"
python3 -c "import transformers ; transformers.AutoConfig.from_pretrained('$MODEL_NAME')"
python3 -c "import transformers ; transformers.AutoTokenizer.from_pretrained('$MODEL_NAME', use_fast=True)"

# Load httpproxy last since it blocks access to HF
! module load httpproxy
python3 -m wandb login
