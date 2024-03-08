import os
import json

import numpy as np
import torch
import pandas as pd

from datetime import datetime
import argparse

import wandb
import ray

import pathlib
from utils import (
    fill_defaults,
    alg_creator,
    create_dataframes,
    get_env_creator,
    plot_with_ci,
)


os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_SILENT"] = "true"

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

parser = argparse.ArgumentParser(
    description="Run Adaptive Prices and Promotions Experiments"
)
parser.add_argument("--config", type=str)
parser.add_argument("--save_path", type=str, default="")
parser.add_argument("--wandb", action="store_true")
args = parser.parse_args()

if len(args.save_path):
    path = f"{args.save_path}/results"
else:
    current_path = pathlib.Path(__file__).parent.absolute()
    path = f"{current_path}/results"

print(f"Saving results to {path}")

torch.multiprocessing.set_sharing_strategy("file_system")


class AlgWrapper:
    def __init__(self, env, alg, env_type, name):
        self.name = name
        self.env = env
        self.alg = alg
        self.env_type = env_type
        self.ys = []
        self.grad_sizes = []


@ray.remote(scheduling_strategy="SPREAD")
def run_single_iteration(r, alg_param, global_params, exp_id):
    np.random.seed()
    torch.seed()
    xs = []

    start_time = datetime.now()
    log_prog_every = global_params["log_prog_every"]
    write_runs = global_params["write_runs"]
    checkpoint = global_params["checkpoint"]

    if args.wandb:
        wandb.init(
            project="pricing",
            name="experiment_run_{}".format(r),
            group="experiment",
            config={"alg": alg_param, **global_params},
        )

    a = fill_defaults(alg_param=alg_param, global_params=global_params)

    env_creator = get_env_creator(global_params=global_params, alg_param=alg_param)

    alg = AlgWrapper(
        env_creator(),
        alg_creator(a["alg_module"], a["alg_class"], a["params"]),
        a["env"],
        a["name"],
    )
    max_T = alg.env.T

    prev = start_time
    for T in range(0, max_T):
        if (T + 1) % log_prog_every == 0:
            now = datetime.now()
            print(
                f"iteration {T + 1} from {max_T} ({(T + 1) / max_T * 100}%) of alg {alg.name} run {r} elapsed {(now - start_time).total_seconds()} {(now - prev).total_seconds() / log_prog_every}"
            )
            prev = now

        d = {}
        additional_information = alg.env.next_step(alg.alg)
        d[f"regret_{alg.name}"] = alg.env.regret.item()
        if "last_grad" in additional_information:
            d[f"grad_{alg.name}"] = additional_information["last_grad"]
        alg.grad_sizes.append(additional_information.get("last_grad", 0))
        alg.ys.append(alg.env.regret.item())
        if args.wandb:
            wandb.log(d, step=T + 1)
        xs.append(T + 1)

        if write_runs and (T + 1) % checkpoint == 0:
            create_dataframes(
                path,
                alg,
                xs,
                r,
                timediff=now - start_time,
                write=write_runs,
                exp_id=exp_id,
            )

    end_time = datetime.now()
    regret_df, grad_df, regret_at_time_df, history_dfs, time_df = create_dataframes(
        path, alg, xs, r, timediff=end_time - start_time, write=False, exp_id=exp_id
    )

    if args.wandb:
        wandb.finish()
    return (regret_df, regret_at_time_df, grad_df, history_dfs, time_df)


if __name__ == "__main__":
    print(args)
    with open(args.config, "r") as f:
        params = json.load(f)
        print(params)

    repeat = params["global"]["repeat"]

    gp = params["global"]
    exp_id = f"{gp['exp_name']}_{gp['K']}_rep{gp['repeat']}_T{gp['max_T']}"

    ray.init(address="auto")

    # Pick out active algs
    params["algs"] = [
        params["algs"][i]
        for i in range(len(params["algs"]))
        if params["algs"][i]["active"]
    ]
    print("ACTIVE ALGS", [x["name"] for x in params["algs"]])

    if not os.path.exists(f"{path}"):
        os.mkdir(f"{path}")
    if not os.path.exists(f"{path}/{exp_id}"):
        os.mkdir(f"{path}/{exp_id}")

    results = ray.get(
        [
            run_single_iteration.remote(
                r, alg_param=alg_param, global_params=params["global"], exp_id=exp_id
            )
            for r in range(repeat)
            for alg_param in params["algs"]
        ]
    )

    dfs = [r[0] for r in results]
    reg_steps = [r[1] for r in results]
    grads = [r[2] for r in results]
    histories = [r[3] for r in results]
    times = [r[4] for r in results]

    history_names = ["choice", "price", "demand", "x", "context"]
    for i, history_name in enumerate(history_names):
        history_df = pd.concat([history[i] for history in histories])
        history_df.to_csv(f"{path}/{exp_id}/{history_name}_history.csv")

    names = [alg["name"] for alg in params["algs"]]
    df = pd.concat(dfs)
    df.reset_index(inplace=True)
    df.to_csv(f"{path}/{exp_id}/regret.csv")
    plot_with_ci(df, "Regret", f"{path}/{exp_id}/regret.png", names, traces=True)

    reg_steps = pd.concat(reg_steps)
    reg_steps.reset_index(inplace=True)
    reg_steps.to_csv(f"{path}/{exp_id}/simpleregret.csv")
    plot_with_ci(
        reg_steps, "Regret", f"{path}/{exp_id}/simpleregret.png", names, log=True
    )

    grads = pd.concat(grads)
    grads.reset_index(inplace=True)
    grads.to_csv(f"{path}/{exp_id}/gradient_size.csv")
    plot_with_ci(
        grads, "GradientNorm", f"{path}/{exp_id}/gradient_size.png", names, log=True
    )

    times_df = pd.concat(times)
    times_df.to_csv(f"{path}/{exp_id}/times.csv")

    with open(f"{path}/{exp_id}/config.json", "w") as outfile:
        json.dump(params, outfile)
