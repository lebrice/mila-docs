# CodingTips - Debugging and Profiling tutorial


## Prerequisites

1. Have SSH access to a SLURM cluster, with SSH access to compute nodes configured.
    - If you have an active job on a compute node, say `cn-g001`, you need to be able to connect to it from your local machine with `ssh cn-g001`. If not, setup a ProxyJump following the [Mila docs](https://docs.mila.quebec/Userguide.html#connecting-to-compute-nodes).
2. Have [UV](https://docs.astral.sh/uv/getting-started/installation) installed on the cluster
   - We also recommend using UV locally.
3. Have `milatools` installed locally (`uv tool install milatools`)


## Getting Started

```bash
ssh mila 'git clone https://github.com/mila-iqia/mila-docs --branch debugging_profiling_tutorial'
mila code mila-docs/docs/examples/advanced/imagenet --alloc --ntasks=2 --nodes=1-2 --gpus-per-task=l40s:1 --cpus-per-task=4 --mem-per-gpu=24G --time=02:59:00 --partition=main
```

In the VSCode terminal, run this:

```bash
uv sync
```

In another terminal, run this:

```bash
uvx --with=torch-tb-profiler tensorboard --logdir $SCRATCH/checkpoints
```


## TODOs

- [ ] This is crashing with a CUDA OOM error after a few steps, why?
- [ ] Figure out why the logging to wandb is not working correctly! Why is it logging the loss only once?


