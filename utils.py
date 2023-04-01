from base.environments.non_contextual import MultipleProbobalisticEnvironment
from base.environments.contextual import MultipleProbobalisticContextualEnvironment

from base.parameter_models import get_parameter_model_class

import m3p.algorithms
import base.algorithms.greedy
import ts.algorithms

from matplotlib import pyplot as plt

import torch
import numpy as np
import pandas as pd
import sys
import os
import ray
import shutil


def alg_creator(alg_module, alg_class, params):
    if "params_model_class" in params:
        params["params_model_class"] = get_parameter_model_class(
            params["params_model_class"]
        )
    try:
        print(alg_module, alg_class)
        C = getattr(sys.modules[alg_module], alg_class)
        return C(**params)
    except Exception as e:
        raise Exception("Unknown Algorithm Type", e)


def get_parameter_model_class(c_str):
    try:
        return getattr(sys.modules["base.parameter_models"], c_str)
    except:
        raise Exception("Unknown Parameter Class")


def create_dataframes(path, alg, xs, r, write, exp_id):
    T = len(xs)

    regret_df = {"Regret": np.array(alg.ys), "T": xs, "run": r, "Algorithm": alg.name}
    regret_df = pd.DataFrame(regret_df)

    grad_df = {
        "GradientNorm": alg.grad_sizes,
        "T": xs,
        "run": r,
        "Algorithm": alg.name,
    }
    grad_df = pd.DataFrame(grad_df)

    regret_at_time_df = {
        "Regret": np.array(alg.ys)[1:] - np.array(alg.ys)[:-1],
        "T": xs[1:],
        "run": r,
        "Algorithm": alg.name,
    }
    regret_at_time_df = pd.DataFrame(regret_at_time_df)

    choice_history_df = pd.DataFrame(
        {
            "T": xs,
            "Algorithm": alg.name,
            "run": r,
            "choices": alg.env.choice_history[:T],
        }
    )
    choice_history_df["choices"] = choice_history_df["choices"].astype(str)
    price_history_df = pd.DataFrame(
        {
            "T": xs,
            "Algorithm": alg.name,
            "run": r,
            "prices": alg.env.price_history[:T, :].numpy().tolist(),
        }
    )
    price_history_df["prices"] = price_history_df["prices"].astype(str)
    demand_history_df = pd.DataFrame(
        {
            "T": xs,
            "Algorithm": alg.name,
            "run": r,
            "demands": alg.env.demand_history[:T, :].detach().numpy().tolist(),
        }
    )
    demand_history_df["demands"] = demand_history_df["demands"].astype(str)
    if alg.env_type in ["NonContextual", "Contextual"]:
        x_history_df = pd.DataFrame(
            {
                "T": xs,
                "Algorithm": alg.name,
                "run": r,
                "xs": alg.env.x_history[:T, :].numpy().tolist(),
            }
        )
        x_history_df["xs"] = x_history_df["xs"].astype(str)
    else:
        x_history_df = pd.DataFrame()
    if alg.env_type == "Contextual":
        context_history_df = pd.DataFrame(
            {
                "T": xs,
                "Algorithm": alg.name,
                "run": r,
                "contexts": alg.env.context_history[:T, :].numpy().tolist(),
            }
        )
        context_history_df["contexts"] = context_history_df["contexts"].astype(str)
    else:
        context_history_df = pd.DataFrame()
    if write:
        output_dir = f"{path}/{exp_id}/runs/{alg.name}_{r}"
        if os.path.exists(output_dir):
            print("removing", output_dir)
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        ds_choice = ray.data.from_pandas(choice_history_df)
        ds_choice.repartition(1).write_csv(f"{output_dir}/choice_history.csv")

        ds_demand = ray.data.from_pandas(demand_history_df)
        ds_demand.repartition(1).write_csv(f"{output_dir}/demand_history.csv")

        ds_price = ray.data.from_pandas(price_history_df)
        ds_price.repartition(1).write_csv(f"{output_dir}/price_history.csv")

        ds_x = ray.data.from_pandas(x_history_df)
        ds_x.repartition(1).write_csv(f"{output_dir}/x_history.csv")
        if alg.env_type == "Contextual":
            ds_context = ray.data.from_pandas(context_history_df)
            ds_context.repartition(1).write_csv(f"{output_dir}/context_history.csv")
        ds_regret = ray.data.from_pandas(regret_df)
        ds_regret.repartition(1).write_csv(f"{output_dir}/regret_run.csv")

        ds_grad = ray.data.from_pandas(grad_df)
        ds_grad.repartition(1).write_csv(f"{output_dir}/gradient_size_run.csv")

        ds_regret_at_time = ray.data.from_pandas(regret_at_time_df)
        ds_regret_at_time.repartition(1).write_csv(
            f"{output_dir}/regret_at_time_run.csv"
        )
    return (
        regret_df,
        grad_df,
        regret_at_time_df,
        (
            choice_history_df,
            price_history_df,
            demand_history_df,
            x_history_df,
            context_history_df,
        ),
    )


def get_env_creator(global_params, alg_param):
    """Get the environment creator for the given algorithm"""
    K = global_params["K"]

    show_plays = global_params["show_plays"]
    max_T = global_params["max_T"]
    u = global_params["u"]
    el = global_params["el"]

    if global_params["costs"] == "random":
        costs = torch.rand(size=(K,)) + 0.1
    elif global_params["costs"] == "nielsen":
        nielsen_n = global_params["nielsen_n"]
        costs = torch.tensor(
            np.load(f"params/nielsen_margins_{nielsen_n}.npy")
        ).float()[:K]
    else:
        costs = torch.zeros(K)
    x_set = torch.cat([torch.zeros([1, K]), torch.eye(K)])

    a = alg_param
    if a["env"] == "Contextual":
        context_size = a["context_size"]
        pm_class = get_parameter_model_class(a["model_class"])
        pm = pm_class(context_size, K, **a["model_init_params"])
        pm.load_state_dict(torch.load(f"torch_models/{a['model_file']}.pkl"))
        env_creator = lambda: MultipleProbobalisticContextualEnvironment(
            context_size=context_size,
            K=K,
            params_model=pm,
            x_set=x_set,
            el=el,
            u=u,
            marginal_costs=costs,
            show_plays=show_plays,
            context_distribution=a["context_distribution"],
            context_distribution_params=a["context_distribution_params"],
            negative_beta_option=global_params["negative_beta_option"],
            T=max_T,
        )
    elif a["env"] == "NonContextual":
        if global_params["dataset"] == "random":
            alphas = torch.rand(size=(K,)) + 2
            betas = torch.rand(size=(K,)) + 0.1
            gammas = torch.zeros(size=(K,))
        elif global_params["dataset"] == "Nielsen":
            nielsen_n = global_params["nielsen_n"]
            alphas = torch.tensor(
                np.load(f"params/nielsen_alphas_{nielsen_n}.npy")
            ).float()[:K]
            betas = (
                torch.tensor(-np.load(f"params/nielsen_betas_{nielsen_n}.npy")).float()[
                    :K
                ]
                / 32
            )
            gammas = torch.tensor(
                np.load(f"params/nielsen_gammas_{nielsen_n}.npy")
            ).float()[:K]
        elif global_params["dataset"] == "simgam0":
            print("JUST REMEMBER, use three products when doing simgam0")
            alphas = torch.tensor([1, 2, 3, 1, 2, 3]).float()[:K]
            betas = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.6, 0.7]).float()[:K]
            gammas = torch.tensor([0.8, 0.5, 0.3, 0.1, 0.1, 0.1]).float()[:K]
        else:
            alphas = betas = gammas = None
        env_creator = lambda: MultipleProbobalisticEnvironment(
            alphas=alphas,
            betas=betas,
            gammas=gammas,
            x_set=x_set,
            el=el,
            u=u,
            marginal_costs=costs,
            show_plays=show_plays,
            T=max_T,
        )
    else:
        raise Exception(f"Env type {a['env']} is not known")
    return env_creator


def fill_defaults(alg_param, global_params):
    a = alg_param
    if "tau" not in a["params"].keys():
        a["params"]["tau"] = global_params["tau"]
    if "batch_size" not in a["params"].keys():
        a["params"]["batch_size"] = global_params.get("batch_size", 1)
    if a["env"] == "Contextual":
        if "model_class" not in a.keys():
            a["model_class"] = global_params["model_class"]
        if "model_file" not in a.keys():
            a["model_file"] = global_params["model_file"]
        if "model_init_params" not in a.keys():
            a["model_init_params"] = global_params["model_init_params"]
        if "context_size" not in a.keys():
            a["context_size"] = global_params["context_size"]
        if "context_distribution" not in a.keys():
            a["context_distribution"] = global_params["context_distribution"]
        if "context_distribution_params" not in a.keys():
            a["context_distribution_params"] = global_params.get(
                "context_distribution_params"
            )
    a["params"]["K"] = global_params["K"]
    return a


def plot_with_ci(df, ylab, save_path, names, log=False, traces=False):
    """Plot data in the given dataframe with confidence intervals.

    Args:
        df (pd.DataFrame): Input dataframe.
        ylab (str): Label for y axis.
        save_path (str): Path to save the figure.
        names (list): List of algorithm names
        log (bool, optional): Whether to plot in log scale. Defaults to False.
        traces (bool, optional): Whether to plot traces for each run. Defaults to False.
    """
    d = df.groupby(["Algorithm", "T"]).agg({ylab: [np.mean, np.median, np.std]})
    d.columns = d.columns.droplevel()
    runs = np.max(df["run"])
    d["ci"] = d["std"] / np.sqrt(runs) * 1.96

    for l, d_p in zip(names, [d.loc[(n)] for n in names]):
        plt.plot(range(1, len(d_p["mean"]) + 1), d_p["mean"], label=l)
        print("d_p", d_p["mean"])
        plt.fill_between(
            range(1, len(d_p["mean"]) + 1),
            d_p["mean"] - d_p["ci"],
            d_p["mean"] + d_p["ci"],
            alpha=0.1,
        )
        if traces:
            for r in range(runs):
                d_r = df[(df["Algorithm"] == l) & (df["run"] == r)]
                t = d_r["T"]
                d_r = d_r.drop(["run", "Algorithm", "T", "index"], axis=1)
                d_r = d_r.reset_index(drop=True)
                print("plotz", t, d_r)
                plt.plot(d_r, alpha=0.05, color=plt.gca().lines[-1].get_color())

        plt.legend()
    if log:
        plt.yscale("log")
    plt.savefig(save_path)
    plt.clf()
