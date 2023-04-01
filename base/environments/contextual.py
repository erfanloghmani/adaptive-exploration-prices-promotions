import numpy as np
from base.models.contextual import (
    MultipleMNLContextualModel,
)
import torch


class MultipleProbobalisticContextualEnvironment:
    def __init__(
        self,
        context_size,
        K,
        params_model,
        x_set,
        marginal_costs,
        el,
        u,
        context_distribution,
        context_distribution_params,
        negative_beta_option,
        T,
        show_plays=False,
    ):
        """Contextual Environment

        Args:
            context_size (int): Context vector dimensionality.
            K (int): Number of products.
            params_model (torch.nn.Module): The model for the alpha, beta, and gamma parameters.
            x_set (torch.tensor): Set of possible promotions, each row contains one promotion scenario on K products.
            marginal_costs (torch.tensor): Marginal costs of each product.
            el (float): Lower bound of prices.
            u (float): Upper bound of prices.
            context_distribution (str): Distribution of the context.
            context_distribution_params (dict): Configuration of the context distribution.
            negative_beta_option (str): How to deal with the negative betas.
            T (int): Time horizon of the experiment.
            show_plays (bool, optional): Whether to show each time step situation and what action the method took. Defaults to False.
        """
        self.context_size = context_size
        self.K = K
        self.T = T
        self.t = 0
        self.regret = 0

        self.price_history = torch.zeros((T, K))
        self.x_history = torch.zeros((T, K))
        self.choice_history = torch.zeros(T, dtype=torch.long)
        self.context_history = torch.zeros((T, context_size))
        self.demand_history = torch.zeros((T, K))

        self.model = MultipleMNLContextualModel(
            context_size,
            K,
            params_model,
            x_set=x_set,
            marginal_costs=marginal_costs,
            el=el,
            u=u,
            negative_beta_option=negative_beta_option,
        )
        self.regret_history = []
        self.show_plays = show_plays
        self.context_distribution = context_distribution
        self.context_distribution_params = context_distribution_params

    def next_step(self, alg):
        """Simulate the next step of the environment.

        Args:
            alg (Algorithm): Algorithm object to be used

        Returns:
            dict: Dictionary containing additional information returned by the algorithm.
        """
        c = self.next_c()
        (next_price, next_x), additional_information = alg.next_price_x(self, c)
        next_price = torch.clip(next_price, min=self.model.el, max=self.model.u)
        self.set_price_x(c, next_price.view(1, -1), next_x, additional_information)
        return additional_information

    def next_c(self):
        """generate the next context vector.

        Raises:
            Exception: When context_distribution is not supported.

        Returns:
            torch.tensor: context vector.
        """
        if self.context_distribution == "ones":
            return torch.ones((1, self.context_size))
        if self.context_distribution == "weighted-averages":
            k = np.random.exponential(scale=1.0, size=self.context_size)
            return torch.tensor(k / sum(k)).view(1, -1).float()
        if self.context_distribution == "box":
            return (torch.rand((1, self.context_size)) + 0.5) / self.context_size
        if self.context_distribution == "orthogonal-groups":
            return (
                torch.eye(self.context_size)[np.random.randint(self.context_size)]
                .reshape(1, -1)
                .float()
            )
        if self.context_distribution == "gmm":
            probs = self.context_distribution_params["probs"]
            n_c = len(probs)
            group = np.random.choice(np.arange(n_c), p=probs)
            group_center = torch.tensor(
                self.context_distribution_params["groups"][group]
            )
            group_draw = group_center + self.context_distribution_params["group_stds"][
                group
            ] * torch.randn_like(group_center)
            return group_draw.view(1, -1).float()
        if self.context_distribution == "store_quarter":
            quarter = self.t // (self.T // 4)
            store_ratios = self.context_distribution_params["store_ratios"]
            ns = len(store_ratios)
            store = np.random.choice(np.arange(ns), p=store_ratios)
            c = torch.zeros((1, 4 + ns))
            c[0, quarter] = 1
            c[0, 4 + store] = 1
            return c
        raise Exception(f"context_distribution {self.context_distribution} not found")

    def set_price_x(self, c, p, x, additional_information={}):
        """Set the prices and promotions of the next step.

        Args:
            c (torch.tensor): Context vector.
            p (torch.tensor): Price vector.
            x (torch.tensor): Promotion vector.
            additional_information (dict, optional): Additional information of the algorithm. Defaults to {}.
        """
        best_price, best_x = self.model.max_mean_demand_price_x(c)
        self.best_revenue = self.model.mean_profit(c, best_price, best_x)

        self.t += 1
        self.context_history[self.t - 1, :] = c
        self.price_history[self.t - 1, :] = p
        self.x_history[self.t - 1, :] = x

        choice = self.model.sample_choice(c, p, x)
        self.choice_history[self.t - 1] = choice
        mean_demand = self.model.mean_demand(c, p, x)
        self.demand_history[self.t - 1, :] = mean_demand

        rev = mean_demand.matmul((p - self.model.marginal_costs).T)[0, 0]
        self.regret += self.best_revenue - rev
        self.regret_history.append(self.regret.item())

        if self.show_plays:
            print(f"iter: {self.t}, c: {c}")
            print(
                f"best_price: {best_price} best_x: {best_x}, best_rev: {self.best_revenue}"
            )
            print(f"played_p: {p} played_x: {x}, rev: {rev}")
