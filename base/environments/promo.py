from base.models.promo import (
    MultipleMNLPromoModel,
)
import torch


class MultipleProbobalisticPromoEnvironment:
    def __init__(
        self,
        alphas,
        betas,
        gammas,
        x_set,
        marginal_costs,
        el,
        u,
        T,
        show_plays=False,
    ):
        """NonContextual Environment

        Args:
            alphas (torch.tensor): Alpha parameters
            betas (torch.tensor): Beta parameters
            gammas (torch.tensor): Gamma parameters
            x_set (torch.tensor): Set of possible promotions, each row contains one promotion scenario on K products.
            marginal_costs (torch.tensor): Marginal costs of each product.
            el (float): Lower bound of prices.
            u (float): Upper bound of prices.
            T (int): Time horizon of the experiment.
            show_plays (bool, optional): Whether to show each time step situation and what action the method took. Defaults to False.
        """
        self.t = 0
        self.T = T
        self.regret = 0
        self.regret_history = []

        K = alphas.shape[0]
        self.price_history = torch.zeros((T, K))
        self.x_history = torch.zeros((T, K))
        self.choice_history = torch.zeros(T, dtype=torch.long)
        self.demand_history = torch.zeros((T, K))

        self.model = MultipleMNLPromoModel(
            alphas,
            betas,
            gammas,
            x_set=x_set,
            marginal_costs=marginal_costs,
            el=el,
            u=u,
        )
        self.best_price, self.best_x = self.model.max_mean_demand_price_x()
        self.best_revenue = self.model.mean_profit(self.best_price, self.best_x)

        self.show_plays = show_plays

    def next_step(self, alg):
        """Simulate the next step of the environment.

        Args:
            alg (Algorithm): Algorithm object to be used

        Returns:
            dict: Dictionary containing additional information returned by the algorithm.
        """
        (next_price, next_x), additional_information = alg.next_price_x(self)
        next_price = torch.clip(next_price, min=self.model.el, max=self.model.u)
        self.set_price_x(next_price, next_x, additional_information)
        return additional_information

    def set_price_x(self, p, x, additional_information={}):
        """Set the prices and promotions of the next step.

        Args:
            p (torch.tensor): Price vector.
            x (torch.tensor): Promotion vector.
            additional_information (dict, optional): Additional information of the algorithm. Defaults to {}.
        """
        self.t += 1

        self.price_history[self.t - 1, :] = p
        self.x_history[self.t - 1, :] = x

        choice = self.model.sample_choice(p, x)
        self.choice_history[self.t - 1] = choice

        mean_demand = self.model.mean_demand(p, x)
        self.demand_history[self.t - 1, :] = mean_demand
        rev = mean_demand.dot(p - self.model.marginal_costs)
        self.regret += self.best_revenue - rev
        self.regret_history.append(self.regret.item())

        if self.show_plays:
            print(f"iter: {self.t}")
            print(
                f"best_price: {self.best_price} best_x: {self.best_x}, best_rev: {self.best_revenue}"
            )
            print(f"played_p: {p} played_x: {x}, rev: {rev}")
