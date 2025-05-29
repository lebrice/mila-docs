LLM Fine-tuning Job
===================

Prerequisites:

* :doc:`/examples/frameworks/pytorch_setup/index`
* :doc:`/examples/distributed/single_gpu/index`
* :doc:`/examples/distributed/multi_gpu/index`
* :doc:`/examples/distributed/multi_node/index`

Other interesting resources:

* `<https://sebarnold.net/dist_blog/>`_
* `<https://lambdalabs.com/blog/multi-node-pytorch-distributed-training-guide>`_
* `<https://huggingface.co/docs/trl/main/en/grpo_trainer#grpo-at-scale-train-a-70b-model-on-multiple-nodes>`_


Click here to see `the code for this example
<https://github.com/mila-iqia/mila-docs/tree/master/docs/examples/distributed/llm_finetuning>`_

**job.sh**

.. literalinclude:: job.sh
    :language: bash

**main.py**

.. literalinclude:: main.py
    :language: python


**Running this example**

1. Install UV from https://docs.astral.sh/uv

2. Launch the job:

.. code-block:: bash

    $ sbatch job.sh
