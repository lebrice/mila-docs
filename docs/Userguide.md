# User's guide

...or *IDT's list of opinionated howtos*

This section seeks to provide users of the Mila infrastructure with practical knowledge, tips and tricks and example commands.

## Quick Start

Users first need [login access to the cluster](#logging_in). It is recommended to install [milatools](https://github.com/mila-iqia/milatools) which will help in the [set up of the ssh configuration](#mila_init) needed to securely and easily connect to the cluster.

### mila code

milatools also makes it easy to run and debug code on the Mila cluster.

First you need to setup your ssh configuration using `mila init`. The initialisation of the ssh configuration is explained [here](#mila_init) and in the [mila init section of github page](https://github.com/mila-iqia/milatools#mila-init).

Once that is done, you may run [VSCode](https://code.visualstudio.com/) on the cluster simply by [using the Remote-SSH extension](https://code.visualstudio.com/docs/remote/ssh#_connect-to-a-remote-host) and selecting `mila-cpu` as the host (in step 2).

`mila-cpu` allocates a single CPU and 8 GB of RAM. If you need more resources from within VSCode (e.g. to run a ML model in a notebook), then you can use `mila code`. For example, if you want a GPU, 32G of RAM and 4 cores, run this command in the terminal:

```bash
mila code path/on/cluster --alloc --gres=gpu:1 --mem=32G -c 4
```

The details of the command can be found in the [mila code section of github page](https://github.com/mila-iqia/milatools#mila-code). Remember that you need to first setup your ssh configuration using `mila init` before the `mila code` command can be used.

---

<!-- Additional sections from the original .rst can be converted and appended here. -->
