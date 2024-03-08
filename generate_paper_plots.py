import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import torch
from base.models.contextual import MultipleMNLContextualModel
from base.environments.contextual import MultipleProbobalisticContextualEnvironment
from base.environments.non_contextual import MultipleProbobalisticEnvironment
from base.parameter_models import *

from base.models.contextual import MultipleMNLContextualModel
from base.parameter_models import ParameterModelLinear
import argparse
import os
import json
from collections import OrderedDict


TS_LABEL = "Thompson Sampling"
GREEDY_LABEL = "Greedy"
M3P_LABEL = "M3P"

LABEL_FONT = 18
TICK_FONT = 12
LEGEND_FONT = 16

rolling_window = 300

DPI = 600


def plot_with_ci(
    df,
    ylab,
    target,
    names,
    to_file,
    max_T,
    log=False,
    logx=False,
    ci=True,
    traces=False,
    rolling=False,
    ylim_top=None,
    ylim_bottom=None,
    grid=False,
    scatter=False,
    metric="mean",
    xlimleft=None,
    plot_cfg={},
    batch_size=100,
    colors=None,
    style=None,
    alpha=1,
    legend_font=LEGEND_FONT,
    tick_font=TICK_FONT,
    label_font=LABEL_FONT,
    legend_loc="best",
    tick_step=5000,
    title=None,
    show_final_line=[],
):
    d = df.groupby(["Algorithm", "T"]).agg({target: [np.mean, np.median, np.std]})
    d.columns = d.columns.droplevel()
    runs = np.max(df["run"]) + 1
    d["ci"] = d["std"] / np.sqrt(runs)  # * 1.96

    for l, d_p, k in zip(
        names.values(), [d.loc[(n)] for n in names.keys()], names.keys()
    ):
        if rolling:
            plt.plot(
                np.arange(1, len(d_p[metric]) + 1) / batch_size,
                d_p[metric].rolling(rolling_window).mean(),
                label=l,
                alpha=0.6 if "M3P" in l else 1,
                c=colors[k],
                linestyle=style[k],
            )
        elif scatter:
            plt.scatter(
                np.arange(1, len(d_p[metric]) + 1) / batch_size,
                d_p[metric],
                label=l,
                alpha=0.08,
                c=colors[k],
                linestyle=style[k],
            )
        else:
            if l in show_final_line:
                plt.axhline(
                    y=d_p[metric].iloc[-1],
                    label=l,
                    c=colors[k],
                    linestyle=style[k],
                )
            else:
                plt.plot(
                    np.arange(1, len(d_p[metric]) + 1) / batch_size,
                    d_p[metric],
                    label=l,
                    c=colors[k],
                    linestyle=style[k],
                    alpha=alpha,
                )
                if ci:
                    plt.fill_between(
                        np.arange(1, len(d_p[metric]) + 1) / batch_size,
                        d_p["mean"] - d_p["ci"],
                        d_p["mean"] + d_p["ci"],
                        alpha=0.1,
                        color=colors[k],
                    )
        if traces:
            for r in range(runs):
                d_r = df[(df["Algorithm"] == l) & (df["run"] == r)]
                t = d_r["T"]
                d_r = d_r.drop(["run", "Algorithm", "T", "index"], axis=1)
                d_r = d_r.reset_index(drop=True)
                plt.plot(d_r.index, d_r[target], alpha=0.4, label=r)
        # leg = plt.legend(fontsize=legend_font, loc=legend_loc)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        leg = plt.legend(
            by_label.values(), by_label.keys(), fontsize=legend_font, loc=legend_loc
        )
        for lh in leg.legendHandles:
            lh.set_alpha(1)
    if log:
        plt.yscale("log")
    if logx:
        plt.xscale("log")
    plt.ylabel(ylab, fontsize=label_font)
    plt.xlabel(r"Time Step $(t)$", fontsize=label_font)
    plt.xticks(np.arange(0, max_T + 1, tick_step), fontsize=tick_font)
    plt.yticks(fontsize=tick_font)

    if title is not None:
        plt.title(title, fontsize=label_font + 2)
    if grid:
        plt.grid()
    if ylim_top is not None:
        plt.ylim(top=ylim_top)
    if ylim_bottom is not None:
        plt.ylim(bottom=ylim_bottom)
    if xlimleft is not None:
        plt.xlim(left=xlimleft)
    plt.xlim([0, max_T])

    plt.savefig(to_file, bbox_inches="tight", dpi=DPI, **plot_cfg)
    print(f"Saved plot to {to_file}")
    plt.show()
    plt.clf()


def generate_plots(
    df,
    a,
    file_suffix,
    env=None,
    batch_size=100,
    colors=None,
    style=None,
    repeat_best_rev=1000,
    max_T=20000,
    percent_ylim_bottom=80,
    file_format="pdf",
    metric="mean",
    style_simple=None,
    only_regret=False,
    result_path="results",
    **kwargs,
):
    if style_simple is None:
        style_simple = style
    plot_with_ci(
        df,
        target="Regret",
        names=a,
        ylab="Regret",
        traces=False,
        to_file=f"{result_path}/paper_plots/{file_suffix}.{file_format}",
        grid=True,
        batch_size=batch_size,
        colors=colors,
        style=style,
        max_T=max_T,
        metric=metric,
        **kwargs,
    )

    if only_regret:
        return

    df_g = df.groupby(["Algorithm", "run"])
    df_appl = df_g.apply(
        lambda x: pd.DataFrame(
            {
                "T": np.arange(0, x.shape[0] - 1),
                "SimpleRegret": np.array(x["Regret"])[1:] - np.array(x["Regret"])[:-1],
            }
        )
    )
    df_sreg = pd.DataFrame(df_appl).reset_index()
    plot_with_ci(
        df_sreg,
        target="SimpleRegret",
        log=True,
        names=a,
        ylab="Simple Regret",
        traces=False,
        to_file=f"{result_path}/paper_plots/simpleregret_{file_suffix}.{file_format}",
        rolling=True,
        grid=True,
        colors=colors,
        style=style_simple,
        batch_size=batch_size,
        max_T=max_T,
        metric=metric,
        **kwargs,
    )
    if isinstance(env, MultipleProbobalisticContextualEnvironment):
        best_revs = []
        for i in range(repeat_best_rev):
            c = env.next_c()
            env.t += 1
            best_price, best_x = env.model.max_mean_demand_price_x(c)
            best_revs.append(
                env.model.mean_profit(c, best_price, best_x).detach().numpy()
            )
        best_rev = np.mean(best_revs)
    else:
        best_rev = env.best_revenue
    df["Max_Rev"] = df["T"].apply(lambda x: x * best_rev)
    df["Regret-Percentage"] = (1 - df["Regret"] / df["Max_Rev"]) * 100
    plot_with_ci(
        df,
        ylim_top=101,
        ylim_bottom=percent_ylim_bottom,
        target="Regret-Percentage",
        names=a,
        ylab=" Percent",
        traces=False,
        to_file=f"{result_path}/paper_plots/paper_plot_percentage_{file_suffix}.{file_format}",
        grid=True,
        colors=colors,
        style=style,
        batch_size=batch_size,
        rolling=False,
        max_T=max_T,
        metric=metric,
        **kwargs,
    )
    return df


x_set_9 = torch.cat([torch.zeros([1, 9]), torch.eye(9)])
costs_9 = torch.zeros(9)


def plots_5_1_noncontextual_3_products_prices(T, segments, result_path, rep):
    exp_name = f"5-1-noncontextual-3-products_3_rep{rep}_T{T}"
    df_3p_prices = pd.read_csv(f"{result_path}/{exp_name}/price_history.csv")
    product_id = 0
    df_3p_prices["price"] = df_3p_prices["prices"].apply(
        lambda ps: json.loads(ps)[product_id]
    )
    for segment in segments:
        segment_sel = (df_3p_prices["T"] > segment[0]) & (
            df_3p_prices["T"] < segment[1]
        )
        data_TS = df_3p_prices[
            df_3p_prices["Algorithm"].str.contains("TS") & segment_sel
        ]["price"]
        data_Greedy = df_3p_prices[
            df_3p_prices["Algorithm"].str.contains("Greedy") & segment_sel
        ]["price"]

        plt.hist(data_Greedy, 50, label=GREEDY_LABEL, alpha=0.7, density=True)
        plt.hist(data_TS, 50, label=TS_LABEL, alpha=0.6, density=True)
        plt.xlim([15, 30])
        plt.ylim([0, 0.6])
        plt.legend(fontsize=LEGEND_FONT)
        plt.xlabel("Price", fontsize=LABEL_FONT)
        plt.xticks(fontsize=TICK_FONT)
        plt.yticks(fontsize=TICK_FONT)
        file_suffix = f"5-1-noncontextual-3-products-prices-{segment[0]}-{segment[1]}"
        to_file = f"{result_path}/paper_plots/{file_suffix}.png"
        plt.savefig(to_file, bbox_inches="tight")
        print(f"Saved plot to {to_file}")
        plt.show()
        plt.clf()


def plots_5_1_noncontextual_3_products_promotions(T, segments, result_path, rep):
    exp_name = f"5-1-noncontextual-3-products_3_rep{rep}_T{T}"
    df_3p_promotions = pd.read_csv(f"{result_path}/{exp_name}/x_history.csv")
    for segment in segments:
        segment_sel = (df_3p_promotions["T"] > segment[0]) & (
            df_3p_promotions["T"] < segment[1]
        )
        width = 0.35
        x = np.arange(4)
        labels = ["Product 1", "Product 2", "Product 3", "No Promotion"]
        fig, ax = plt.subplots()
        for i, alg in enumerate(["Greedy", "TS"]):
            label = TS_LABEL if alg == "TS" else GREEDY_LABEL
            played = np.mean(
                df_3p_promotions[
                    df_3p_promotions["Algorithm"].str.contains(alg) & segment_sel
                ]["xs"].apply(lambda x: np.fromstring(x[1:-1], sep=","))
            )
            played = np.append(played, 1 - np.sum(played))
            ax.bar(x - (-1) ** i * width / 2, played, width, alpha=1, label=label)

        ax.set_ylabel("Proportion", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=14)
        ax.legend(fontsize=LEGEND_FONT)
        ax.set_aspect(1.4)
        ax.set_ylim([0, 1.1])
        file_suffix = (
            f"5-1-noncontextual-3-products-promotions-{segment[0]}-{segment[1]}"
        )
        to_file = f"{result_path}/paper_plots/{file_suffix}.png"
        plt.savefig(to_file, bbox_inches="tight")
        print(f"Saved plot to {to_file}")
        plt.show()
        plt.clf()


def plots_5_1_noncontextual_3_products(T, result_path, segments=None, rep=40):
    exp_name = f"5-1-noncontextual-3-products_3_rep{rep}_T{T}"
    df_3p = pd.read_csv(f"{result_path}/{exp_name}/regret.csv")
    legend_3p_noncontext = {
        "TS-Laplace-Batch-10-lr0.0001": TS_LABEL,
        "Greedy-Batch-10-lr0.0001": GREEDY_LABEL,
    }
    colors_3p_noncontext = {
        "TS-Laplace-Batch-10-lr0.0001": "C1",
        "Greedy-Batch-10-lr0.0001": "C0",
    }
    style_3p_noncontext = {
        "TS-Laplace-Batch-10-lr0.0001": "-",
        "Greedy-Batch-10-lr0.0001": "-",
    }
    print(df_3p["Algorithm"].unique())

    K = 3
    alphas = torch.tensor([1, 2, 3, 1, 2, 3]).float()[:K]
    betas = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.6, 0.7]).float()[:K]
    gammas = torch.tensor([0.8, 0.5, 0.3, 0.1, 0.1, 0.1]).float()[:K]
    env = MultipleProbobalisticEnvironment(
        alphas=alphas,
        betas=betas,
        gammas=gammas,
        x_set=torch.cat([torch.zeros([1, K]), torch.eye(K)]),
        el=0,
        u=40,
        marginal_costs=torch.zeros(K),
        show_plays=False,
        T=1000,
    )

    generate_plots(
        df=df_3p,
        a=legend_3p_noncontext,
        env=env,
        file_suffix=exp_name,
        batch_size=1,
        colors=colors_3p_noncontext,
        style=style_3p_noncontext,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=80,
        result_path=result_path,
    )
    if segments is None:
        segments = [(0, 2000), (10000, 20000), (40000, 50000)]
    plots_5_1_noncontextual_3_products_prices(T, segments, result_path, rep)
    plots_5_1_noncontextual_3_products_promotions(T, segments, result_path, rep)


def plots_5_1_noncontextual_3_products_promo_check(T, result_path):
    exp_name = f"5-1-noncontextual-3-products_3_rep40_T{T}"
    df_3p = pd.read_csv(f"{result_path}/{exp_name}/regret.csv")
    legend_3p_noncontext = {
        "TS-Laplace-Batch-10-lr0.0001": TS_LABEL,
        "TS-Laplace-Batch-10-lr0.0001-xrandom": TS_LABEL + " random x",
        "Greedy-Batch-10-lr0.0001": GREEDY_LABEL,
    }
    colors_3p_noncontext = {
        "TS-Laplace-Batch-10-lr0.0001-xrandom": "C3",
        "TS-Laplace-Batch-10-lr0.0001": "C1",
        "Greedy-Batch-10-lr0.0001": "C0",
    }
    style_3p_noncontext = {
        "TS-Laplace-Batch-10-lr0.0001": "-",
        "TS-Laplace-Batch-10-lr0.0001-xrandom": "-",
        "Greedy-Batch-10-lr0.0001": "-",
    }
    print(df_3p["Algorithm"].unique())
    styles = ["-", "--", "-.", ":"]
    for i in range(2, 3):
        if i == -1:
            k = 0
        else:
            k = 2**i
        kb_str = f"{k:03b}"
        kb_str = ", ".join(kb_str)
        legend_3p_noncontext[f"TS-Laplace-Batch-10-lr0.0001-x{k:03b}"] = (
            TS_LABEL + rf" $x = {kb_str}$"
        )
        colors_3p_noncontext[f"TS-Laplace-Batch-10-lr0.0001-x{k:03b}"] = "C2"
        style_3p_noncontext[f"TS-Laplace-Batch-10-lr0.0001-x{k:03b}"] = styles[i + 1]

    K = 3
    alphas = torch.tensor([1, 2, 3, 1, 2, 3]).float()[:K]
    betas = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.6, 0.7]).float()[:K]
    gammas = torch.tensor([0.8, 0.5, 0.3, 0.1, 0.1, 0.1]).float()[:K]
    env = MultipleProbobalisticEnvironment(
        alphas=alphas,
        betas=betas,
        gammas=gammas,
        x_set=torch.cat([torch.zeros([1, K]), torch.eye(K)]),
        el=0,
        u=40,
        marginal_costs=torch.zeros(K),
        show_plays=False,
        T=1000,
    )

    generate_plots(
        df=df_3p,
        a=legend_3p_noncontext,
        env=env,
        file_suffix=exp_name + "-promo-check",
        batch_size=1,
        colors=colors_3p_noncontext,
        style=style_3p_noncontext,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=80,
        tick_step=10000,
        result_path=result_path,
    )


def plots_5_1_noncontextual_3_products_expthencommit(T, result_path):
    exp_name = f"5-1-noncontextual-3-products-expthencommit_3_rep40_T{T}"
    df_3p = pd.read_csv(f"{result_path}/{exp_name}/regret.csv")
    print(df_3p["Algorithm"].unique())

    K = 3
    alphas = torch.tensor([1, 2, 3, 1, 2, 3]).float()[:K]
    betas = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.6, 0.7]).float()[:K]
    gammas = torch.tensor([0.8, 0.5, 0.3, 0.1, 0.1, 0.1]).float()[:K]
    env = MultipleProbobalisticEnvironment(
        alphas=alphas,
        betas=betas,
        gammas=gammas,
        x_set=torch.cat([torch.zeros([1, K]), torch.eye(K)]),
        el=0,
        u=40,
        marginal_costs=torch.zeros(K),
        show_plays=False,
        T=1000,
    )

    def generate_plots_for_tau(tau):
        legend_3p_noncontext = {
            f"TS-Laplace-tau-{tau}-Batch-10-lr0.0001": TS_LABEL,
            f"ExpThenCommit-tau-{tau}-Batch-10-lr0.0001": "Explore-then-Commit",
        }
        colors_3p_noncontext = {
            f"TS-Laplace-tau-{tau}-Batch-10-lr0.0001": "C1",
            f"ExpThenCommit-tau-{tau}-Batch-10-lr0.0001": "C0",
        }
        style_3p_noncontext = {
            f"TS-Laplace-tau-{tau}-Batch-10-lr0.0001": "-",
            f"ExpThenCommit-tau-{tau}-Batch-10-lr0.0001": "-",
        }
        generate_plots(
            df=df_3p,
            a=legend_3p_noncontext,
            env=env,
            file_suffix=exp_name + f"tau-{tau}",
            batch_size=1,
            colors=colors_3p_noncontext,
            style=style_3p_noncontext,
            file_format="png",
            max_T=T,
            percent_ylim_bottom=80,
            tick_step=10000,
            # title=rf"$\tau = {tau}$",
            only_regret=True,
            result_path=result_path,
        )

    generate_plots_for_tau(10)
    generate_plots_for_tau(100)
    generate_plots_for_tau(500)
    generate_plots_for_tau(1000)
    generate_plots_for_tau(10000)


def plots_5_2_noncontextual_nielsen_batch_200(T, env_nielsen, result_path):
    exp_name = f"5-2-noncontextual-nielsen-batch-200-langevin-1_9_rep40_T{T}"
    df_nielsen = pd.read_csv(f"{result_path}/{exp_name}/regret.csv")

    legend_nielsen_noncontext = {
        "TS-Laplace-Batch-200-lr0.1": "TS-Laplace Batch 200",
        "TS-Langevin-Batch-200-lr0.01": "TS-Langevin Batch 200",
        "Greedy-Batch-200-lr0.1": "Greedy Batch 200",
        "M3P-lr0.1": "M3P",
    }
    colors_nielsen_noncontext = {
        "TS-Laplace-Batch-200-lr0.1": "green",
        "TS-Langevin-Batch-200-lr0.01": "red",
        "Greedy-Batch-200-lr0.1": "blue",
        "M3P-lr0.1": "orange",
    }
    style_nielsen_noncontext = {
        "TS-Laplace-Batch-200-lr0.1": "-",
        "TS-Langevin-Batch-200-lr0.01": "-",
        "Greedy-Batch-200-lr0.1": "-",
        "M3P-lr0.1": "-",
    }

    generate_plots(
        df=df_nielsen,
        a=legend_nielsen_noncontext,
        env=env_nielsen,
        file_suffix=exp_name,
        batch_size=1,
        colors=colors_nielsen_noncontext,
        style=style_nielsen_noncontext,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=85,
        result_path=result_path,
    )
    return df_nielsen


def plots_5_2_noncontextual_nielsen_batch_200_grid(T, result_path):
    return plots_5_2_noncontextual_nielsen_batch_grid(
        T, batch=200, result_path=result_path
    )


def plots_5_2_noncontextual_nielsen_batch_10_grid(T, result_path):
    return plots_5_2_noncontextual_nielsen_batch_grid(
        T, batch=10, result_path=result_path
    )


def plots_5_2_noncontextual_nielsen_batch_200_grid_greedy(T, result_path):
    batch = 200
    K = 9
    nielsen_n = "9"
    alphas = torch.tensor(np.load(f"params/nielsen_alphas_{nielsen_n}.npy")).float()[:K]
    betas = (
        torch.tensor(-np.load(f"params/nielsen_betas_{nielsen_n}.npy")).float()[:K] / 32
    )
    gammas = torch.tensor(np.load(f"params/nielsen_gammas_{nielsen_n}.npy")).float()[:K]

    env_nielsen = MultipleProbobalisticEnvironment(
        alphas=alphas,
        betas=betas,
        gammas=gammas,
        x_set=x_set_9,
        el=0,
        u=40,
        marginal_costs=costs_9,
        show_plays=False,
        T=1000,
    )

    exp_name = f"5-2-noncontextual-nielsen-batch-{batch}-greedy_9_rep40_T{T}"
    df_nielsen = pd.read_csv(f"{result_path}/{exp_name}/regret.csv")
    print(df_nielsen["Algorithm"].unique())
    legend_nielsen_noncontext = {
        "Greedy-Batch-200-lr0.1-100steps": "100 gradient steps",
        "Greedy-Batch-200-lr0.1-200steps": "200 gradient steps",
        "Greedy-Batch-200-lr0.1-500steps": "500 gradient steps",
    }
    colors_nielsen_noncontext = {
        "Greedy-Batch-200-lr0.1-100steps": "blue",
        "Greedy-Batch-200-lr0.1-200steps": "blue",
        "Greedy-Batch-200-lr0.1-500steps": "blue",
    }
    style_nielsen_noncontext = {
        "Greedy-Batch-200-lr0.1-100steps": "-",
        "Greedy-Batch-200-lr0.1-200steps": "--",
        "Greedy-Batch-200-lr0.1-500steps": "-.",
    }

    generate_plots(
        df=df_nielsen,
        a=legend_nielsen_noncontext,
        env=env_nielsen,
        file_suffix=exp_name + "_grid_greedy",
        batch_size=1,
        colors=colors_nielsen_noncontext,
        style=style_nielsen_noncontext,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=80,
        only_regret=True,
        result_path=result_path,
    )


def plots_5_2_noncontextual_nielsen_batch_200_grid_line(T, result_path):
    batch = 200
    K = 9
    nielsen_n = "9"
    alphas = torch.tensor(np.load(f"params/nielsen_alphas_{nielsen_n}.npy")).float()[:K]
    betas = (
        torch.tensor(-np.load(f"params/nielsen_betas_{nielsen_n}.npy")).float()[:K] / 32
    )
    gammas = torch.tensor(np.load(f"params/nielsen_gammas_{nielsen_n}.npy")).float()[:K]

    env_nielsen = MultipleProbobalisticEnvironment(
        alphas=alphas,
        betas=betas,
        gammas=gammas,
        x_set=x_set_9,
        el=0,
        u=40,
        marginal_costs=costs_9,
        show_plays=False,
        T=1000,
    )

    exp_name = f"5-2-noncontextual-nielsen-batch-{batch}-grid_9_rep40_T{T}"
    df_nielsen = pd.read_csv(f"{result_path}/{exp_name}/regret.csv")
    exp_name_lr = f"5-2-noncontextual-nielsen-batch-{batch}-grid-lr_9_rep40_T{T}"
    df_lr = pd.read_csv(f"{result_path}/{exp_name_lr}/regret.csv")
    df_lr = df_lr[df_lr["Algorithm"].isin(["Greedy-Batch-200-lr0.1", "M3P-lr0.1"])]
    df_nielsen = pd.concat([df_nielsen, df_lr])
    print(df_nielsen["Algorithm"].unique())
    legend_nielsen_noncontext = {
        "Greedy-Batch-200-lr0.1": "Greedy",
        "M3P-lr0.1": "M3P",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.125": r"TS-Laplace",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.25": r"TS-Laplace",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.5": r"TS-Laplace",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.75": r"TS-Laplace",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp1": r"TS-Laplace",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.125": r"TS-Langevin",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.25": r"TS-Langevin",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.5": r"TS-Langevin",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.75": r"TS-Langevin",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp1": r"TS-Langevin",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.125": r"TS-Langevin",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.25": r"TS-Langevin",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.5": r"TS-Langevin",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.75": r"TS-Langevin",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp1": r"TS-Langevin",
    }
    colors_nielsen_noncontext = {
        "Greedy-Batch-200-lr0.1": "blue",
        "M3P-lr0.1": "orange",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.125": "green",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.25": "green",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.5": "green",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.75": "green",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp1": "green",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.125": "red",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.25": "red",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.5": "red",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.75": "red",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp1": "red",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.125": "red",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.25": "red",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.5": "red",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.75": "red",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp1": "red",
    }
    style_nielsen_noncontext = {
        "Greedy-Batch-200-lr0.1": "-",
        "M3P-lr0.1": "-",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.125": "--",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.25": "--",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.5": "--",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.75": "--",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp1": "--",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.125": "--",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.25": "--",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.5": "--",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.75": "--",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp1": "--",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.125": "--",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.25": "--",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.5": "--",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.75": "--",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp1": "--",
    }

    generate_plots(
        df=df_nielsen,
        a=legend_nielsen_noncontext,
        env=env_nielsen,
        file_suffix=exp_name + "_grid_TS_line",
        batch_size=1,
        colors=colors_nielsen_noncontext,
        style=style_nielsen_noncontext,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=80,
        only_regret=True,
        alpha=0.6,
        show_final_line=["Greedy", "M3P"],
        result_path=result_path,
    )


def plots_5_2_noncontextual_nielsen_batch_grid(T, batch, result_path):
    K = 9
    nielsen_n = "9"
    alphas = torch.tensor(np.load(f"params/nielsen_alphas_{nielsen_n}.npy")).float()[:K]
    betas = (
        torch.tensor(-np.load(f"params/nielsen_betas_{nielsen_n}.npy")).float()[:K] / 32
    )
    gammas = torch.tensor(np.load(f"params/nielsen_gammas_{nielsen_n}.npy")).float()[:K]

    env_nielsen = MultipleProbobalisticEnvironment(
        alphas=alphas,
        betas=betas,
        gammas=gammas,
        x_set=x_set_9,
        el=0,
        u=40,
        marginal_costs=costs_9,
        show_plays=False,
        T=1000,
    )

    exp_name = f"5-2-noncontextual-nielsen-batch-{batch}-grid_9_rep40_T{T}"
    df_nielsen = pd.read_csv(f"{result_path}/{exp_name}/regret.csv")
    print(df_nielsen["Algorithm"].unique())
    legend_nielsen_noncontext = {
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.125": r"TS-Laplace $a = 0.125$",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.25": r"TS-Laplace $a = 0.25$",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.5": r"TS-Laplace $a = 0.5$",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.75": r"TS-Laplace $a = 0.75$",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp1": r"TS-Laplace $a = 1$",
    }
    colors_nielsen_noncontext = {
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.125": "green",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.25": "green",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.5": "green",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.75": "green",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp1": "green",
    }
    style_nielsen_noncontext = {
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.125": (0, (1, 10)),
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.25": "-",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.5": "--",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp0.75": ":",
        f"TS-Laplace-Batch-{batch}-lr0.1-exp1": "-.",
    }

    generate_plots(
        df=df_nielsen,
        a=legend_nielsen_noncontext,
        env=env_nielsen,
        file_suffix=exp_name + "_grid_TS_Laplace",
        batch_size=1,
        colors=colors_nielsen_noncontext,
        style=style_nielsen_noncontext,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=80,
        only_regret=True,
        result_path=result_path,
    )

    legend_nielsen_noncontext = {
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.125": r"TS-Langevin $\psi = 0.125$",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.25": r"TS-Langevin $\psi = 0.25$",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.5": r"TS-Langevin $\psi = 0.5$",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.75": r"TS-Langevin $\psi = 0.75$",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp1": r"TS-Langevin $\psi = 1$",
    }
    colors_nielsen_noncontext = {
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.125": "red",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.25": "red",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.5": "red",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.75": "red",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp1": "red",
    }
    style_nielsen_noncontext = {
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.125": (0, (1, 10)),
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.25": "-",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.5": "--",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.75": ":",
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp1": "-.",
    }

    generate_plots(
        df=df_nielsen,
        a=legend_nielsen_noncontext,
        env=env_nielsen,
        file_suffix=exp_name + "_grid_TS_Langevin-N20",
        batch_size=1,
        colors=colors_nielsen_noncontext,
        style=style_nielsen_noncontext,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=80,
        only_regret=True,
        result_path=result_path,
    )

    legend_nielsen_noncontext = {
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.125": r"TS-Langevin $\psi = 0.125$",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.25": r"TS-Langevin $\psi = 0.25$",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.5": r"TS-Langevin $\psi = 0.5$",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.75": r"TS-Langevin $\psi = 0.75$",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp1": r"TS-Langevin $\psi = 1$",
    }
    colors_nielsen_noncontext = {
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.125": "darkred",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.25": "darkred",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.5": "darkred",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.75": "darkred",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp1": "darkred",
    }
    style_nielsen_noncontext = {
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.125": (0, (1, 10)),
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.25": "-",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.5": "--",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.75": ":",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp1": "-.",
    }

    generate_plots(
        df=df_nielsen,
        a=legend_nielsen_noncontext,
        env=env_nielsen,
        file_suffix=exp_name + "_grid_TS_Langevin-N50",
        batch_size=1,
        colors=colors_nielsen_noncontext,
        style=style_nielsen_noncontext,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=80,
        only_regret=True,
        result_path=result_path,
    )

    legend_nielsen_noncontext = {
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.5": r"TS-Langevin $N_t = 20$",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.5": r"TS-Langevin $N_t = 50$",
    }
    colors_nielsen_noncontext = {
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.5": "red",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.5": "darkred",
    }
    style_nielsen_noncontext = {
        f"TS-Langevin-Batch-{batch}-lr0.01-N20-exp0.5": "--",
        f"TS-Langevin-Batch-{batch}-lr0.01-N50-exp0.5": "--",
    }

    generate_plots(
        df=df_nielsen,
        a=legend_nielsen_noncontext,
        env=env_nielsen,
        file_suffix=exp_name + "_grid_TS_Langevin-N20-N50",
        batch_size=1,
        colors=colors_nielsen_noncontext,
        style=style_nielsen_noncontext,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=80,
        only_regret=True,
        result_path=result_path,
    )


def plots_5_2_noncontextual_nielsen_batch_200_grid_lr(T, result_path):
    plots_5_2_noncontextual_nielsen_batch_grid_lr(T, 200, result_path)


def plots_5_2_noncontextual_nielsen_batch_10_grid_lr(T, result_path):
    plots_5_2_noncontextual_nielsen_batch_grid_lr(T, 10, result_path)


def plots_5_2_noncontextual_nielsen_batch_grid_lr(T, batch, result_path):
    K = 9
    nielsen_n = "9"
    alphas = torch.tensor(np.load(f"params/nielsen_alphas_{nielsen_n}.npy")).float()[:K]
    betas = (
        torch.tensor(-np.load(f"params/nielsen_betas_{nielsen_n}.npy")).float()[:K] / 32
    )
    gammas = torch.tensor(np.load(f"params/nielsen_gammas_{nielsen_n}.npy")).float()[:K]

    env_nielsen = MultipleProbobalisticEnvironment(
        alphas=alphas,
        betas=betas,
        gammas=gammas,
        x_set=x_set_9,
        el=0,
        u=40,
        marginal_costs=costs_9,
        show_plays=False,
        T=1000,
    )

    exp_name = f"5-2-noncontextual-nielsen-batch-{batch}-grid-lr_9_rep40_T{T}"
    df_nielsen = pd.read_csv(f"{result_path}/{exp_name}/regret.csv")
    print(df_nielsen["Algorithm"].unique())
    method_colors = {
        "Greedy": "blue",
        "M3P": "orange",
        "TS-Laplace": "green",
        "TS-Langevin-20": "red",
        "TS-Langevin-50": "red",
    }
    for method, color in method_colors.items():
        legend_nielsen_noncontext = {}
        colors_nielsen_noncontext = {}
        style_nielsen_noncontext = {}
        styles = ["-", "--", "-.", ":"]
        batch_str = f"-Batch-{batch}" if method != "M3P" else ""

        for i, c in enumerate([1, 0.1, 0.01, 0.001]):
            method_use = method
            divide_str = ""
            if method in ["Greedy", "M3P"]:
                after_lr = ""
            elif method == "TS-Laplace":
                after_lr = f"-exp0.5"
            else:
                after_lr = f"-N20-exp0.5" if "20" in method else f"-N50-exp0.5"
                method_use = "TS-Langevin"
                divide_str = " / t"
            legend_nielsen_noncontext[f"{method_use}{batch_str}-lr{c}{after_lr}"] = (
                rf"{method_use} $\eta_t = {c}{divide_str}$"
            )
            colors_nielsen_noncontext[f"{method_use}{batch_str}-lr{c}{after_lr}"] = (
                color
            )
            style_nielsen_noncontext[f"{method_use}{batch_str}-lr{c}{after_lr}"] = (
                styles[i]
            )

        if method in ["Greedy", "M3P"]:
            for i, c in enumerate([1, 0.1, 0.01, 0.001]):
                legend_nielsen_noncontext[f"{method}{batch_str}-lr{c}-decay"] = (
                    rf"{method} $\eta_t = {c} / t$"
                )
                colors_nielsen_noncontext[f"{method}{batch_str}-lr{c}-decay"] = (
                    f"light{color}"
                    if method == "Greedy"
                    else "darkgoldenrod" if method == "M3P" else color
                )
                style_nielsen_noncontext[f"{method}{batch_str}-lr{c}-decay"] = styles[i]

        generate_plots(
            df=df_nielsen,
            a=legend_nielsen_noncontext,
            env=env_nielsen,
            file_suffix=exp_name + f"_grid_lr_{method}",
            batch_size=1,
            colors=colors_nielsen_noncontext,
            style=style_nielsen_noncontext,
            file_format="png",
            max_T=T,
            percent_ylim_bottom=80,
            only_regret=True,
            ylim_top=30000 if method != "TS-Laplace" else None,
            ylim_bottom=0,
            result_path=result_path,
        )


def plots_5_2_noncontextual_nielsen_batch_10(T, env_nielsen, result_path):
    exp_name = f"5-2-noncontextual-nielsen-batch-10_9_rep40_T{T}"
    df_nielsen = pd.read_csv(f"{result_path}/{exp_name}/regret.csv")

    legend_nielsen_noncontext = {
        "TS-Langevin-Batch-10-lr0.01-N50": "TS-Langevin Batch 10",
        "TS-Laplace-Batch-10-lr0.1": "TS-Laplace Batch 10",
        "Greedy-Batch-10-lr1": "Greedy Batch 10",
    }
    colors_nielsen_noncontext = {
        "TS-Langevin-Batch-10-lr0.01-N50": "red",
        "TS-Laplace-Batch-10-lr0.1": "green",
        "Greedy-Batch-10-lr1": "blue",
    }
    style_nielsen_noncontext = {
        "TS-Langevin-Batch-10-lr0.01-N50": "--",
        "TS-Laplace-Batch-10-lr0.1": "--",
        "Greedy-Batch-10-lr1": "--",
    }

    generate_plots(
        df=df_nielsen,
        a=legend_nielsen_noncontext,
        env=env_nielsen,
        file_suffix="5-2-noncontextual-nielsen-batch-10",
        batch_size=1,
        colors=colors_nielsen_noncontext,
        style=style_nielsen_noncontext,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=90,
        result_path=result_path,
    )
    return df_nielsen


def plots_5_2_noncontextual_nielsen_batch_10_greedy(
    T, df_nielsen, env_nielsen, result_path
):
    legend_nielsen_greedy = {
        "Greedy-Batch-10-lr1": "Greedy Batch 10",
        "Greedy-Batch-200-lr0.1": "Greedy Batch 200",
    }
    colors_nielsen_greedy = {
        "Greedy-Batch-10-lr1": "blue",
        "Greedy-Batch-200-lr0.1": "blue",
    }
    style_nielsen_greedy = {
        "Greedy-Batch-10-lr1": "--",
        "Greedy-Batch-200-lr0.1": "-",
    }

    generate_plots(
        df=df_nielsen,
        a=legend_nielsen_greedy,
        env=env_nielsen,
        file_suffix="5-2-noncontextual-nielsen-batch-10-greedy",
        batch_size=1,
        colors=colors_nielsen_greedy,
        style=style_nielsen_greedy,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=80,
        only_regret=True,
        result_path=result_path,
    )


def plots_5_2_noncontextual_nielsen_batch_10_ts(
    T, df_nielsen, env_nielsen, result_path
):
    legend_nielsen_ts = {
        "TS-Langevin-Batch-10-lr0.01-N50": "TS-Langevin Batch 10",
        "TS-Laplace-Batch-10-lr0.1": "TS-Laplace Batch 10",
        "TS-Langevin-Batch-200-lr0.01": "TS-Langevin Batch 200",
        "TS-Laplace-Batch-200-lr0.1": "TS-Laplace Batch 200",
    }
    colors_nielsen_ts = {
        "TS-Langevin-Batch-10-lr0.01-N50": "red",
        "TS-Laplace-Batch-10-lr0.1": "green",
        "TS-Langevin-Batch-200-lr0.01": "red",
        "TS-Laplace-Batch-200-lr0.1": "green",
    }
    style_nielsen_ts = {
        "TS-Langevin-Batch-10-lr0.01-N50": "--",
        "TS-Laplace-Batch-10-lr0.1": "--",
        "TS-Langevin-Batch-200-lr0.01": "-",
        "TS-Laplace-Batch-200-lr0.1": "-",
    }

    generate_plots(
        df=df_nielsen,
        a=legend_nielsen_ts,
        env=env_nielsen,
        file_suffix="5-2-noncontextual-nielsen-batch-10-ts",
        batch_size=1,
        colors=colors_nielsen_ts,
        style=style_nielsen_ts,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=80,
        only_regret=True,
        result_path=result_path,
    )


def plots_5_2_noncontextual_nielsen(T, result_path):
    K = 9
    nielsen_n = "9"
    alphas = torch.tensor(np.load(f"params/nielsen_alphas_{nielsen_n}.npy")).float()[:K]
    betas = (
        torch.tensor(-np.load(f"params/nielsen_betas_{nielsen_n}.npy")).float()[:K] / 32
    )
    gammas = torch.tensor(np.load(f"params/nielsen_gammas_{nielsen_n}.npy")).float()[:K]

    env_nielsen = MultipleProbobalisticEnvironment(
        alphas=alphas,
        betas=betas,
        gammas=gammas,
        x_set=x_set_9,
        el=0,
        u=40,
        marginal_costs=costs_9,
        show_plays=False,
        T=1000,
    )
    df_nielsen_10 = plots_5_2_noncontextual_nielsen_batch_10(
        T, env_nielsen, result_path
    )
    df_nielsen_200 = plots_5_2_noncontextual_nielsen_batch_200(
        T, env_nielsen, result_path
    )
    df_nielsen_full = pd.concat([df_nielsen_10, df_nielsen_200])
    plots_5_2_noncontextual_nielsen_batch_10_greedy(
        T, df_nielsen_full, env_nielsen, result_path
    )
    plots_5_2_noncontextual_nielsen_batch_10_ts(
        T, df_nielsen_full, env_nielsen, result_path
    )


def plots_7_1_contextual_linear_context(T, context_name, result_path):
    exp_name = f"7-1-contextual-linear-{context_name}_9_rep20_T{T}"
    df_context_lin = pd.read_csv(f"{result_path}/{exp_name}/regret.csv")

    pm = ParameterModelLinear(K=9, context_size=4)
    pm.load_state_dict(torch.load("torch_models/Linear-4in-9*3out.pkl"))

    env = MultipleProbobalisticContextualEnvironment(
        params_model=pm,
        K=9,
        context_size=4,
        context_distribution=context_name,
        negative_beta_option="max",
        x_set=x_set_9,
        marginal_costs=costs_9,
        el=0,
        u=3,
        context_distribution_params={},
        T=1000,
    )
    legend_linear_context = {
        "TS-Langevin": TS_LABEL,
        "Greedy": GREEDY_LABEL,
        "M3P": M3P_LABEL,
    }
    colors_linear_context = {
        "TS-Langevin": "red",
        "Greedy": "blue",
        "M3P": "orange",
    }
    style_linear_context = {
        "TS-Langevin": "-",
        "Greedy": "-",
        "M3P": "-",
    }
    generate_plots(
        df=df_context_lin,
        a=legend_linear_context,
        env=env,
        file_suffix=exp_name,
        batch_size=1,
        colors=colors_linear_context,
        style=style_linear_context,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=70,
        result_path=result_path,
    )
    return df_context_lin, env


def plots_7_1_contextual_linear_context_nc(
    T, context_name, df_context_lin, env, result_path
):
    legend_linear_context = {
        "TS-Langevin": "TS Contextual",
        "TS-Langevin-NC": "TS Non-Contextual",
    }
    colors_linear_context = {
        "TS-Langevin": "red",
        "TS-Langevin-NC": "red",
    }
    style_linear_context = {
        "TS-Langevin": "-",
        "TS-Langevin-NC": "--",
    }
    generate_plots(
        df=df_context_lin,
        a=legend_linear_context,
        env=env,
        file_suffix=f"7-1-contextual-linear-{context_name}_9_rep20_T{T}_nc",
        batch_size=1,
        colors=colors_linear_context,
        style=style_linear_context,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=70,
        only_regret=True,
        result_path=result_path,
    )


def plots_7_1_contextual_linear_context_prices_orthogonal(
    T, env, segments, result_path
):
    exp_name = f"7-1-contextual-linear-orthogonal-groups_9_rep20_T{T}"
    df_prices = pd.read_csv(f"{result_path}/{exp_name}/price_history.csv")
    product_id = 6
    df_prices["price"] = df_prices["prices"].apply(lambda x: json.loads(x)[product_id])

    df_contexts = pd.read_csv(f"results/{exp_name}/context_history.csv")
    context_sel = df_contexts[df_contexts["contexts"] == "[1.0, 0.0, 0.0, 0.0]"]
    lines = []
    context = torch.zeros((1, 4))
    context[0, 0] = 1
    p, x = env.model.max_mean_demand_price_x(context)
    lines.append(p[0, 6].item())

    max_x = 30
    min_x = 5
    max_y = 0.9
    for segment in segments:
        df_prices_s = df_prices.iloc[context_sel.index, :]
        segment_sel = (df_prices_s["T"] > segment[0]) & (df_prices_s["T"] < segment[1])
        data_TS = df_prices_s[
            (df_prices_s["Algorithm"] == "TS-Langevin") & segment_sel
        ]["price"]
        data_Greedy = df_prices_s[(df_prices_s["Algorithm"] == "Greedy") & segment_sel][
            "price"
        ]

        plt.vlines(lines, ymin=0, ymax=max_y, color="green", alpha=0.3, label="Optimum")
        plt.hist(
            data_Greedy,
            80,
            range=[min_x, max_x],
            label=GREEDY_LABEL,
            alpha=0.3,
            density=True,
        )
        plt.hist(
            data_TS, 80, range=[min_x, max_x], label=TS_LABEL, alpha=0.3, density=True
        )
        plt.xlim([min_x, max_x])
        plt.ylim([0, max_y])
        plt.legend(fontsize=LEGEND_FONT)
        plt.xlabel("Price", fontsize=LABEL_FONT)
        plt.xticks(fontsize=TICK_FONT)
        plt.yticks(fontsize=TICK_FONT)

        file_suffix = (
            f"7-1-contextual-linear-orthogonal-groups-prices-{segment[0]}-{segment[1]}"
        )
        to_file = f"{result_path}/paper_plots/{file_suffix}.png"
        plt.savefig(to_file, bbox_inches="tight")
        print(f"Saved plot to {to_file}")
        plt.clf()


def plots_7_1_contextual_linear_context_prices_weighted(T, env, segments, result_path):
    exp_name = f"7-1-contextual-linear-weighted-averages_9_rep20_T{T}"
    df_prices = pd.read_csv(f"{result_path}/{exp_name}/price_history.csv")
    product_id = 6
    df_prices["price"] = df_prices["prices"].apply(lambda x: json.loads(x)[product_id])

    lines = []
    for c in range(1000):
        context = env.next_c()
        p, x = env.model.max_mean_demand_price_x(context)
        lines.append(p[0, product_id].item())

    max_x = 30
    min_x = 5
    max_y = 0.9
    for segment in segments:
        df_prices_s = df_prices
        segment_sel = (df_prices_s["T"] > segment[0]) & (df_prices_s["T"] < segment[1])
        data_TS = df_prices_s[
            (df_prices_s["Algorithm"] == "TS-Langevin") & segment_sel
        ]["price"]
        data_Greedy = df_prices_s[(df_prices_s["Algorithm"] == "Greedy") & segment_sel][
            "price"
        ]

        plt.hist(
            lines,
            80,
            range=[min_x, max_x],
            color="green",
            alpha=0.3,
            label="Optimum",
            density=True,
        )
        plt.hist(
            data_Greedy,
            80,
            range=[min_x, max_x],
            label=GREEDY_LABEL,
            alpha=0.3,
            density=True,
        )
        plt.hist(
            data_TS, 80, range=[min_x, max_x], label=TS_LABEL, alpha=0.3, density=True
        )
        plt.xlim([min_x, max_x])
        plt.ylim([0, max_y])
        plt.legend(fontsize=LEGEND_FONT)
        plt.xlabel("Price", fontsize=LABEL_FONT)
        plt.xticks(fontsize=TICK_FONT)
        plt.yticks(fontsize=TICK_FONT)

        file_suffix = (
            f"7-1-contextual-linear-weighted-averages-prices-{segment[0]}-{segment[1]}"
        )
        to_file = f"{result_path}/paper_plots/{file_suffix}.png"
        plt.savefig(to_file, bbox_inches="tight")
        print(f"Saved plot to {to_file}")
        plt.clf()


def plots_7_1_contextual_linear(T, result_path):
    for context_name in ["orthogonal-groups", "weighted-averages", "box"]:
        df, env = plots_7_1_contextual_linear_context(T, context_name)
        plots_7_1_contextual_linear_context_nc(
            T,
            context_name,
            df_context_lin=df,
            env=env,
            result_path=result_path,
        )
        segments = [(0, 1000), (5000, 6000), (19000, 20000)]
        if context_name == "orthogonal-groups":
            plots_7_1_contextual_linear_context_prices_orthogonal(
                T=T,
                env=env,
                segments=segments,
                result_path=result_path,
            )
        elif context_name == "weighted-averages":
            plots_7_1_contextual_linear_context_prices_weighted(
                T=T,
                env=env,
                segments=segments,
                result_path=result_path,
            )


def experiment_7_2_regret_comparison(T, df, legend):
    def calc_improve(target):
        d = df.groupby(["Algorithm", "T"]).agg({target: [np.mean, np.median, np.std]})
        d.columns = d.columns.droplevel()
        legend_inv = {value: key for key, value in legend.items()}
        greedy_mean = d.loc[(legend_inv[GREEDY_LABEL], T - 1), "mean"]
        ts_mean = d.loc[(legend_inv[TS_LABEL], T - 1), "mean"]
        print(
            f"TS has {(greedy_mean - ts_mean) / greedy_mean * 100:0.2f}% less {target} compared to Greedy"
        )

    calc_improve("Regret")
    calc_improve("Regret-Percentage")


def plots_7_2_contextual_nielsen(T, result_path):
    exp_name = f"7-2-contextual-nielsen-quarter-store_9_rep10_T{T}"
    df_nielsen = pd.read_csv(f"{result_path}/{exp_name}/regret.csv")
    pm = ParameterModelLinear(K=9, context_size=6)
    pm.load_state_dict(torch.load("torch_models/Nielsen-6in-quarter-store-9*3out.pkl"))
    env_9_store = MultipleProbobalisticContextualEnvironment(
        params_model=pm,
        K=9,
        context_size=6,
        context_distribution="store_quarter",
        negative_beta_option="max",
        x_set=x_set_9,
        marginal_costs=costs_9,
        el=0,
        u=3,
        context_distribution_params={"store_ratios": [0.65, 0.35]},
        T=1000,
    )
    legend_nielsen_context = {
        "TS-Langevin": TS_LABEL,
        "Greedy": GREEDY_LABEL,
        "M3P": M3P_LABEL,
    }
    colors_nielsen_context = {
        "TS-Langevin": "red",
        "Greedy": "blue",
        "M3P": "orange",
    }
    style_nielsen_context = {
        "TS-Langevin": "-",
        "Greedy": "-",
        "M3P": "-",
    }

    df = generate_plots(
        df=df_nielsen,
        a=legend_nielsen_context,
        env=env_9_store,
        file_suffix=exp_name,
        batch_size=1,
        colors=colors_nielsen_context,
        style=style_nielsen_context,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=75,
        result_path=result_path,
    )

    experiment_7_2_regret_comparison(T, df, legend_nielsen_context)

    del legend_nielsen_context["M3P"]
    env_9_store = MultipleProbobalisticContextualEnvironment(
        params_model=pm,
        K=9,
        context_size=6,
        context_distribution="store_quarter",
        negative_beta_option="max",
        x_set=x_set_9,
        marginal_costs=costs_9,
        el=0,
        u=3,
        context_distribution_params={"store_ratios": [0.65, 0.35]},
        T=1000,
    )

    generate_plots(
        df=df_nielsen,
        a=legend_nielsen_context,
        env=env_9_store,
        file_suffix=f"{exp_name}-no-m3p",
        batch_size=1,
        colors=colors_nielsen_context,
        style=style_nielsen_context,
        file_format="png",
        max_T=T,
        result_path=result_path,
    )


def plots_7_3_contextual_nonlinear(T, result_path):
    pm = NearestCenter(K=9, context_size=4, n_groups=8)
    pm.load_state_dict(
        torch.load("torch_models/NearestCenter_4in-8c-9*3out-random.pkl")
    )

    conf = json.load(open("configs/7-3-contextual-nonlinear-8groups.json"))
    env_9_nonlin = MultipleProbobalisticContextualEnvironment(
        params_model=pm,
        K=9,
        context_size=4,
        context_distribution="gmm",
        negative_beta_option="max",
        x_set=x_set_9,
        marginal_costs=costs_9,
        el=0,
        u=3,
        context_distribution_params=conf["global"]["context_distribution_params"],
        T=T,
    )
    exp_name = f"7-3-contextual-nonlinear-8groups_9_rep16_T{T}"
    df_nonlin_8c = pd.read_csv(f"{result_path}/{exp_name}/regret.csv")
    legend_nonlin = {
        "TS-Langevin-NotShared": TS_LABEL,
        "Greedy-NotShared": GREEDY_LABEL,
        "M3P-NotShared": M3P_LABEL,
    }
    colors_nonlin = {
        "TS-Langevin-NotShared": "red",
        "Greedy-NotShared": "blue",
        "M3P-NotShared": "orange",
    }
    style_nonlin = {
        "TS-Langevin-NotShared": "-",
        "Greedy-NotShared": "-",
        "M3P-NotShared": "-",
    }

    generate_plots(
        df=df_nonlin_8c,
        a=legend_nonlin,
        env=env_9_nonlin,
        file_suffix=exp_name,
        batch_size=1,
        colors=colors_nonlin,
        style=style_nonlin,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=70,
        result_path=result_path,
    )

    legend_nonlin_with_ts = {
        "TS-Langevin-Linear": "TS Linear",
        "TS-Langevin-Shared": "TS NN Shared Hidden",
        "TS-Langevin-NotShared": "TS NN Non-Shared Hidden",
    }
    colors_nonlin_with_ts = {
        "TS-Langevin-Linear": "darkred",
        "TS-Langevin-Shared": "red",
        "TS-Langevin-NotShared": "red",
    }
    style_nonlin_with_ts = {
        "TS-Langevin-Linear": "--",
        "TS-Langevin-Shared": ":",
        "TS-Langevin-NotShared": "-",
    }

    style_nonlin_with_ts_simple = {
        "TS-Langevin-Linear": "--",
        "TS-Langevin-Shared": (0, (1, 10)),
        "TS-Langevin-NotShared": "-",
    }

    generate_plots(
        df=df_nonlin_8c,
        a=legend_nonlin_with_ts,
        env=env_9_nonlin,
        file_suffix=f"{exp_name}-ts",
        batch_size=1,
        colors=colors_nonlin_with_ts,
        style=style_nonlin_with_ts,
        file_format="png",
        max_T=T,
        percent_ylim_bottom=50,
        style_simple=style_nonlin_with_ts_simple,
        result_path=result_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for the paper")
    parser.add_argument("-experiment", type=str)
    parser.add_argument("-result_path", type=str, default="results")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    result_path = args.result_path

    experiment_map = {
        "5-1": plots_5_1_noncontextual_3_products,
        "5-1-promo": plots_5_1_noncontextual_3_products_promo_check,
        "5-1-expthencommit": plots_5_1_noncontextual_3_products_expthencommit,
        "5-2": plots_5_2_noncontextual_nielsen,
        "5-2-grid200": plots_5_2_noncontextual_nielsen_batch_200_grid,
        "5-2-grid200-greedy": plots_5_2_noncontextual_nielsen_batch_200_grid_greedy,
        "5-2-grid200-line": plots_5_2_noncontextual_nielsen_batch_200_grid_line,
        "5-2-grid200-lr": plots_5_2_noncontextual_nielsen_batch_200_grid_lr,
        "5-2-grid10": plots_5_2_noncontextual_nielsen_batch_10_grid,
        "5-2-grid10-lr": plots_5_2_noncontextual_nielsen_batch_10_grid_lr,
        "7-1": plots_7_1_contextual_linear,
        "7-2": plots_7_2_contextual_nielsen,
        "7-3": plots_7_3_contextual_nonlinear,
    }
    T_map = {
        "5-1": 50000,
        "5-1-promo": 50000,
        "5-1-expthencommit": 50000,
        "5-2": 20000,
        "5-2-grid200": 20000,
        "5-2-grid200-greedy": 10000,
        "5-2-grid200-line": 20000,
        "5-2-grid200-lr": 20000,
        "5-2-grid10": 20000,
        "5-2-grid10-lr": 20000,
        "7-1": 20000,
        "7-2": 40000,
        "7-3": 40000,
    }
    os.makedirs(f"{result_path}/paper_plots", exist_ok=True)
    if not args.all:
        experiment_fun = experiment_map[args.experiment]
        experiment_fun(T_map[args.experiment], result_path=result_path)
    else:
        for exp_id, experiment_fun in experiment_map.items():
            experiment_fun(T_map[exp_id], result_path=result_path)
