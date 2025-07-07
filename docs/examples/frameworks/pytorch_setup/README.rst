.. NOTE: This file is auto-generated from examples/frameworks/pytorch_setup/index.rst
.. This is done so this file can be easily viewed from the GitHub UI.
.. **DO NOT EDIT**

.. _pytorch_setup:

PyTorch Setup
=============

**Prerequisites**: (Make sure to read the following before using this example!)


The full source code for this example is available on `the mila-docs GitHub
repository.
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/frameworks/pytorch_setup>`_


* :ref:`Quick Start`
* :ref:`Running your code`
* :ref:`uv`


**job.sh**


.. code:: bash

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


**pyproject.toml**

.. code:: toml

   [project]
   name = "pytorch-setup"
   version = "0.1.0"
   description = "Add your description here"
   readme = "README.md"
   requires-python = ">=3.12"
   dependencies = [
       "numpy>=2.3.1",
       "torch>=2.7.1",
   ]


**main.py**

.. code:: python

   import torch
   import torch.backends.cuda


   def main():
       cuda_built = torch.backends.cuda.is_built()
       cuda_avail = torch.cuda.is_available()
       device_count = torch.cuda.device_count()

       print(f"PyTorch built with CUDA:         {cuda_built}")
       print(f"PyTorch detects CUDA available:  {cuda_avail}")
       print(f"PyTorch-detected #GPUs:          {device_count}")
       if device_count == 0:
           print("    No GPU detected, not printing devices' names.")
       else:
           for i in range(device_count):
               print(f"    GPU {i}:      {torch.cuda.get_device_name(i)}")


   if __name__ == "__main__":
       main()


**Running this example**

This assumes that you already installed UV on the cluster you are working on.
To create this environment, we first request resources for an interactive job.
Note that we are requesting a GPU for this job, even though we're only going to
install packages. This is because we want PyTorch to be installed with GPU
support, and to have all the required libraries.

.. code-block:: bash

    $ salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00
    salloc: --------------------------------------------------------------------------------------------------
    salloc: # Using default long partition
    salloc: --------------------------------------------------------------------------------------------------
    salloc: Pending job allocation 2959785
    salloc: job 2959785 queued and waiting for resources
    salloc: job 2959785 has been allocated resources
    salloc: Granted job allocation 2959785
    salloc: Waiting for resource configuration
    salloc: Nodes cn-g022 are ready for job
    $ # Load anaconda
    $ module load anaconda/3
    $ # Create the environment (see the example):
    $ conda create -n pytorch python=3.9 pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    (...)
    $ # Press 'y' to accept if everything looks good.
    (...)
    $ # Activate the environment:
    $ conda activate pytorch

Exit the interactive job once the environment has been created. Then, the
example can be launched to confirm that everything works:

.. code-block:: bash

    $ sbatch job.sh
