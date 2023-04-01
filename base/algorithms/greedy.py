import numpy as np
import torch
from base.models.non_contextual import MultipleMNLModel
from base.models.contextual import MultipleMNLContextualModel


class Greedy:
    def __init__(
        self,
        tau,
        K,
        batch_size,
        regularization_factor_mle,
        mle_method,
        mle_recycle,
        mle_lr,
        mle_steps,
    ):
        """Greedy Algorithm For Prices and Promotions (Non-Contextual)

        Args:
            tau (int): Number of initial exploration rounds
            K (int): Number of products
            batch_size (int): Batch size
            regularization_factor_mle (float): Regulatization factor for MLE
            mle_method (string): MLE method
            mle_recycle (boolean): Whether to recycle the parameters from previous step for MLE
            mle_lr (float): Learning rate of MLE
            mle_steps (int): Number of steps of MLE
        """
        self.tau = tau
        self.batch_size = batch_size
        self.regularization_factor_mle = regularization_factor_mle
        self.alpha_bar = None
        self.mle_method = mle_method
        self.mle_recycle = mle_recycle
        self.mle_lr = mle_lr
        self.mle_steps = mle_steps
        self.last_grad = None

    def next_price_x(self, env, c=None):
        """What price to play at the current state of the environment

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
        if (env.t - self.tau) % self.batch_size == 0:
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
                alphas_s_old=self.alpha_bar,
                betas_s_old=self.beta_bar,
                gammas_s_old=self.gamma_bar,
                method=self.mle_method,
                recycle=self.mle_recycle,
                lr=self.mle_lr,
                regularization=self.regularization_factor_mle,
                steps=self.mle_steps,
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


class ContextualGreedy:
    def __init__(
        self,
        tau,
        K,
        batch_size,
        regularization_factor_mle,
        params_model_class,
        params_model_class_config,
        mle_method,
        mle_recycle,
        mle_lr,
        mle_steps,
    ):
        """Contextual Version of Greedy Algorithm For Prices and Promotion

        Args:
            tau (int): Number of initial exploration rounds
            K (int): Number of products
            batch_size (int): Batch size
            regularization_factor_mle (float): Regulatization factor for MLE
            params_model_class (Class): Parameter model class, e.g., ParameterModelLinear
            params_model_class_config (dict): Parameters passed to init method of Parameter Model Class
            mle_method (string): MLE method
            mle_recycle (boolean): Whether to recycle the parameters from previous step for MLE
            mle_lr (float): Learning rate of MLE
            mle_steps (int): Number of steps of MLE
        """
        self.tau = tau
        self.batch_size = batch_size
        self.regularization_factor_mle = regularization_factor_mle
        self.alpha_bar = None
        self.params_model_class = params_model_class
        self.params_model_class_config = params_model_class_config
        self.mle_method = mle_method
        self.mle_recycle = mle_recycle
        self.mle_lr = mle_lr
        self.params_model_bar = None
        self.last_grad = None
        self.mle_steps = mle_steps

    def next_price_x(self, env, c):
        """What price to play at the current state of the environment

        Args:
            env (Environment): Contextual environment
            c (torch.array): Context vector

        Returns:
            tuple: A tuple of optimal action (price, promotion), and additional data
        """
        if env.t < self.tau:
            p = (
                torch.rand(size=(1, env.model.K)) * (env.model.u - env.model.el)
                + env.model.el
            )
            x = env.model.x_set[np.random.randint(env.model.x_set.shape[0])]
            return (p, x), {}
        if (env.t - self.tau) % self.batch_size == 0:
            self.params_model_bar, self.last_grad = MultipleMNLContextualModel.mle(
                Cs=env.context_history[: env.t],
                Is=env.choice_history[: env.t],
                Ps=env.price_history[: env.t],
                Xs=env.x_history[: env.t],
                K=env.model.K,
                context_size=env.model.context_size,
                params_model_class=self.params_model_class,
                method=self.mle_method,
                recycle=self.mle_recycle,
                steps=self.mle_steps,
                lr=self.mle_lr,
                regularization=self.regularization_factor_mle,
                params_model_class_config=self.params_model_class_config,
                params_model_s_old=self.params_model_bar,
            )
        model_res = MultipleMNLContextualModel(
            env.model.context_size,
            env.model.K,
            self.params_model_bar,
            x_set=env.model.x_set,
            marginal_costs=env.model.marginal_costs,
            negative_beta_option=env.model.negative_beta_option,
            el=env.model.el,
            u=env.model.u,
        )
        data = {}
        if self.last_grad is not None:
            data["last_grad"] = self.last_grad
        return model_res.max_mean_demand_price_x(c), data
