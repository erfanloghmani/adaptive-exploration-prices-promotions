# Effective Adaptive Exploration of Prices and Promotions in Choice-Based Demand Models

This is the official repository containing implementations of the Thompson Sampling methods for adaptively setting prices and promotions proposed in the paper. Additionally, the code contains the baselines of Greedy, and M3P method. The repository contains code and instructions for reproducing the experimental results in the paper.

## News

- 🔥🔥🔥 [Feb 2024] The paper got accepted by the INFORMS Marketing Science Journal
- 🔥🔥🔥 [June 2023] Prof. Lalit Jain presented the paper at Summer Institute in Competitive Strategy (SICS) 2023 UC Berkeley

## Citation

If you find this repository useful for your research, please consider citing the following paper.

```
@article{jain2023effective,
  title={Effective Adaptive Exploration of Prices and Promotions in Choice-Based Demand Models},
  author={Jain, Lalit and Li, Zhaoqi and Loghmani, Erfan and Mason, Blake and Yoganarasimhan, Hema},
  journal={Available at SSRN 4438537},
  year={2023}
}
```

## Highlights

1. The proposed method simultaneously incorporates pricing and promotions in a multinomial choice model.
2. The method could also incorporate customer heterogeneity.
3. The code is easy to setup and can handle batched data to reduce the runtime. Additionally, the code uses a computationally efficient model for finding optimal prices and promotions.
4. Using `ray`, the code can easily run in a distributed mode using multiple compute machines.

## Prerequisites

This repo is tested with Python 3.10.8 and PyTorch 1.13.0 in Ubuntu 20.04.1 environment. To install the dependencies use the following command inside your virtual environment.

```bash
pip install -r requirements.txt
```

## Quickstart

This code bases utilizes `ray` for multiprocessing and distributed computing. You could run on single machine or multiple machines. Here is how to run the code on a single machine.

Start with running a ray process:

```bash
ray start --head
```

Now you could run the experiments using the following command

```bash
python main.py --config {path/to/config/file}
```

where you should replace the `{path/to/config/file}` with the path to the config file. Directory [configs](configs/) contains seperate config files for each of the experiments, along with the description about the structure of the config files.

For instance, if you want to run the non-contextual experiment on the nielsen data with batch size of 200, you should use

```bash
python main.py --config configs/5-2-noncontextual-nielsen-batch-200.json
```

## Tutorial

The file [tutorial.ipynb](tutorial.ipynb) contains a tutorial on how to use the code, run the experiments, and generate the plots.

## WandB

If you want to use [wandb](https://wandb.ai/), you should pass `--wandb` as an argument:

```bash
python main.py --config {path/to/config/file} --wandb
```

Note that you should do

```bash
wandb init
```

before running the experiment (only once for the first time) to set up the wandb.

## Generating plots

To generate plots you can use the `generate_paper_plots.py`, after running the experiments. You should either provide what experiment you want the plots for, or use `--all` argument. To specify the experiment you should run

```bash
python generate_paper_plots.py -experiment {EXPERIMENT}
```

where the `{EXPERIMENT}` could be any of `5-1`, `5-2`, `7-1`, `7-2`, `7-3`.

## Multiple Machines

To run on multiple machines, you should first set up virtual environments on each machine. Then, on one of the machines run:

```bash
ray start --head
```

This command will give you instructions with the ip address of the machine you have chosen as your head machine. To connect other machines you should run

```bash
ray start --address='{HEAD_NODE_IP}:6379'
```

on each of the other machines, and `{HEAD_NODE_IP}` is the ip address of the head machine.

You could view the ray dashboard at port `8265` on the head machine.

To run the code in this setting, you should use

```bash
ray job submit --address='{HEAD_NODE_IP}:6379' --working-dir . -- python main.py --config {path/to/config/file} 
```

If your config file has `write_runs=true` you should also pass `--save_path {path/to/results/dir}` where results directory should be a directory shared by all of the machines (Possibly a mounted cloud storage). When you submit the ray job, it gives you a SUBMISSION_ID that you could use to stop the job.

```bash
ray job stop {SUBMISSION_ID}
```
