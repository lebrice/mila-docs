#!/bin/bash
#SBATCH --nodes=5
#SBATCH --gres=gpu:8

# TODOS:
# - Adapt this sbatch script:
#    - [ ] Use a total of 4 gpus (for example in the short-unkillable partition).
#          To start with, use 1 gpu for VLLM, and 3 gpus for training. Nothing too fancy.
#    - [ ] Use a smaller model and use a small number of steps so the full example can
#          run in a reasonable time (e.g. 10 minutes).


# Get the list of allocated nodes
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))

# Assign the first 4 nodes for training and the 5th node for vLLM
TRAIN_NODES="${NODELIST[@]:0:4}"  # Nodes 0, 1, 2, 3 for training
VLLM_NODE="${NODELIST[4]}"  # Node 4 for vLLM

# Run training on the first 4 nodes (Group 1)
srun --nodes=4 --ntasks=4 --nodelist="${NODELIST[@]:0:4}" accelerate launch \
     --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
     --num_processes 32 \
     --num_machines 4 \
     --main_process_ip ${NODELIST[0]} \
     --machine_rank $SLURM_PROCID \
     --rdzv_backend c10d \
     train_grpo.py \
     --server_ip $VLLM_NODE &

# Run vLLM server on the 5th node (Group 2)
srun --nodes=1 --ntasks=1 --nodelist="${NODELIST[4]}" trl vllm-serve --model Qwen/Qwen2.5-72B --tensor_parallel_size 8 &

wait