# CodingTips - Debugging and Profiling tutorial


## Prerequisites

1. Have SSH access to a SLURM cluster, with SSH access to compute nodes configured.
    - If you have an active job on a compute node, say `cn-g001`, you need to be able to connect to it from your local machine with `ssh cn-g001`. 
    - If not, follow the instructions in the Mila docs here: https://docs.mila.quebec/Userguide.html#connecting-to-compute-nodes

2. Have [UV](https://docs.astral.sh/uv/getting-started/installation) installed on the cluster
   - We also recommend using UV locally.

3. Have `milatools` installed locally (run `uv tool install milatools` on your laptop)

4. Clone this repository on the Mila cluster:

```bash
ssh mila 'git clone https://github.com/mila-iqia/mila-docs --branch with_bugs'
```


## Getting Started

1. Get an interactive job with at least 2 GPUS (could be on one or two nodes):

    From a login node:
    ```bash
    salloc --ntasks=2 --nodes=1-2 --gpus-per-task=l40s:1 --mem=24G --time=02:59:00 --partition=main
    ```

- NOTE: For this tutorial, we do it in two steps. Later, you can use mila code to do it in one step like so:

    ```bash
    (local)$  mila code mila-docs/docs/examples/advanced/imagenet --alloc --ntasks=2 --nodes=1-2 --gpus-per-task=l40s:1 --cpus-per-task=4 --mem-per-gpu=24G --time=02:59:00 --partition=main
    ```


2. Open VsCode on the compute node where you got the interactive job (e.g. `cn-l001`):

    ```bash
    (local)$  mila code mila-docs/docs/examples/advanced/imagenet --node=cn-l001
    ```

    You can also use the `"Remote-SSH: Connect to Host..."` action in VsCode, it does the same thing.

3. In the interactive job terminal (salloc, not integrated terminal of vscode)

    ```bash
    cd ~/mila-docs/docs/examples/advanced/imagenet
    uv sync
    srun --ntasks-per-node=1 uv run python prepare_data.py
    ```

4. In VsCode, use the `"Python: Select interpreter"` action, with `"Enter interpreter path..."` and select the .venv at `~/mila-docs/.venv/bin/python`.


5. In VsCode, open the integrated terminal (Ctrl+\`) and run:

```bash
mkdir -p $SCRATCH/checkpoints
uv run tensorboard --logdir $SCRATCH/checkpoints
```

## Mission 1 - Checkpointing Trouble

### Context:
- You want to start running jobs on the Tamia cluster with its shiny H100s. The maximum job length is 24 hours, but your jobs need more time! You added checkpointing to your jobs, but…
(OR):
- In order to get more results in the same amount of time, you decide to start submitting your jobs in multiple incremental chunks of 3h with checkpointing. You jobs now get scheduled very fast! However…

**How can you check if checkpointing is working correctly?**

**Your objectives:**
1. Use the command-line arguments of the example and the vscode debugger to inspect the checkpointing portion of a distributed job.
2. Check that the model weights and data batches are exactly the same given the same seed.
3. Check that when resuming from a checkpoint, the first batch of the next epoch is the same as if there had been no preemption. 


## Mission 2 - Hunting for Bottlenecks

### Context
You received an email from IDT or IT-support or DRAC (formerly ComputeCanada) support, telling you that your job is not making a very good use of its GPUs! You GPU utilization 

**You thought you were doing your best! You’re not sure where to start! What can you make better? And How?**

### Your mission

Identify and fix the causes of poor gpu utilization to **achieve at least 95% gpu utilization (with >90% sm_efficiency)** and then **improve the training speed by at least 2x**.

Steps:
1. Use command-line arguments of the example in combination with the Tensorboard viewer to create an efficient profiling setup, with a feedback loop of less than one minute.
2. Use simple sanity checks to determine if dataloading is the bottleneck
Identify problematic areas of the code (tip: unnecessary Cuda synchronize events)
1. Use the profiler to determine if <X> helps improve performance, and if so, by how much?
    X = using torch.compile
    X = using mixed precision training

Tip: Use `--epochs=1 --limit_train_samples=10000 --limit_val_samples=2000` to keep runs short.
