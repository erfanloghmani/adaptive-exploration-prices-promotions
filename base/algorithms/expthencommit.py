import numpy as np
import torch
from base.models.non_contextual import MultipleMNLModel


class ExpThenCommit:
    def __init__(
        self,
        tau,
        K,
        batch_size,
        regularization_factor_mle,
        mle_method,
        mle_lr,
        mle_steps,
    ):
        """ExpThenCommit Algorithm For Prices and Promotions (Non-Contextual)

        Args:
            tau (int): Number of initial exploration rounds
            K (int): Number of products
            regularization_factor_mle (float): Regulatization factor for MLE
            mle_method (string): MLE method
            mle_lr (float): Learning rate of MLE
            mle_steps (int): Number of steps of MLE
        """
        self.tau = tau
        self.batch_size = batch_size
        self.regularization_factor_mle = regularization_factor_mle
        self.alpha_bar = None
        self.mle_method = mle_method
        self.mle_lr = mle_lr
        self.mle_steps = mle_steps
        self.last_grad = None

    def next_price_x(self, env, c=None):
        """What price to play at the current state of the environment
        Play a random price in the first tau rounds, then estimate a model play the optimal price according to the model for the rest of the rounds

        Args:
            env (Environment): Environment (could be contextual or not)
            c (torch.array, optional): Placeholder for context which is not used for this non-contextual method

        Returns:
            tuple: A tuple of optimal action (price, promotion), and additional data
        """
        if env.t < self.tau:
            p = torch.rand(env.model.K) * (env.model.u - env.model.el) + env.model.el
            x = env.model.x_set[np.random.randint(env.model.x_set.shape[0])]
            self.alpha_bar, self.beta_bar, self.gamma_bar = None, None, None
            return (p, x), {}
        if (env.t - self.tau) == 0:
            mle_steps = int(self.mle_steps * (self.tau / self.batch_size))
            (
                self.alpha_bar,
                self.beta_bar,
                self.gamma_bar,
                self.last_grad,
            ) = MultipleMNLModel.mle(
                Is=env.choice_history[: env.t],
                Ps=env.price_history[: env.t],
                Xs=env.x_history[: env.t],
                K=env.model.K,
                recycle=False,
                method=self.mle_method,
                lr=self.mle_lr,
                regularization=self.regularization_factor_mle,
                steps=mle_steps,
            )
        model_res = MultipleMNLModel(
            self.alpha_bar,
            self.beta_bar,
            self.gamma_bar,
            x_set=env.model.x_set,
            marginal_costs=env.model.marginal_costs,
        )
        data = (
            {
                "alpha_bar": self.alpha_bar,
                "beta_bar": self.beta_bar,
                "gamma_bar": self.gamma_bar,
            }
            if self.alpha_bar is not None
            else {}
        )
        if self.last_grad is not None:
            data["last_grad"] = self.last_grad
        return (
            model_res.max_mean_demand_price_x(),
            data,
        )
