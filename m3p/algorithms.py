import torch
from base.models.non_contextual import MultipleMNLModel
from base.models.contextual import MultipleMNLContextualModel
import numpy as np


class M3P:
    def __init__(
        self, K, regularization_factor_mle, mle_method, mle_recycle, mle_lr, **kwargs
    ):
        """M3P for NonContextual Prices and Promotions Experiments.

        Args:
            K (int): Number of products
            regularization_factor_mle (float): Regulatization factor for MLE
            mle_method (string): MLE method
            mle_recycle (boolean): Whether to recycle the parameters from previous step for MLE
            mle_lr (float): Learning rate of MLE
        """
        self.k = 1
        self.current_idx = 0
        self.current_steps = self.k + K
        self.K = K
        self.regularization_factor_mle = regularization_factor_mle
        self.alpha_bar = None
        self.mle_method = mle_method
        self.mle_recycle = mle_recycle
        self.mle_lr = mle_lr
        self.last_grad = None

    def next_price_x(self, env, c=None):
        """What price to play at the current state of the environment

        Args:
            env (Environment): Environment (could be contextual or not)
            c (torch.array, optional): Placeholder for context which is not used for this non-contextual method

        Returns:
            tuple: A tuple of optimal action (price, promotion), and additional data
        """
        if self.current_idx < self.K:
            p = torch.rand(self.K) * (env.model.u - env.model.el) + env.model.el
            x = env.model.x_set[np.random.randint(env.model.x_set.shape[0])]
            self.alpha_bar, self.beta_bar, self.gamma_bar = None, None, None
        else:
            if self.current_idx == self.K:
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
                )
            model_res = MultipleMNLModel(
                self.alpha_bar,
                self.beta_bar,
                self.gamma_bar,
                x_set=env.model.x_set,
                marginal_costs=env.model.marginal_costs,
            )
            p, x = model_res.max_mean_demand_price_x()
        self.current_idx += 1
        if self.current_idx == self.current_steps:
            self.k += 1
            self.current_idx = 0
            self.current_steps = self.k + self.K
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
        return (p, x), data


class ContextualM3P:
    def __init__(
        self,
        K,
        params_model_class,
        params_model_class_config,
        regularization_factor_mle,
        mle_method,
        mle_recycle,
        mle_lr,
        mle_steps,
        exploration_mult,
        **kwargs
    ):
        """Contextual M3P for Prices and Promotions Experiments

        Args:
            K (int): Number of products
            params_model_class (Class): Parameter model class, e.g., ParameterModelLinear
            params_model_class_config (dict): Parameters passed to init method of Parameter Model Class.
            regularization_factor_mle (float): Regulatization factor for MLE
            mle_method (string): MLE method
            mle_recycle (boolean): Whether to recycle the parameters from previous step for MLE
            mle_lr (float): Learning rate of MLE
            mle_steps (int): Number of MLE steps
            exploration_mult (int): We do exploration_mult times K exploration steps at each iteration of M3P,
                                    it should be set to dimension of the context in contextual settings.
        """

        self.k = 1
        self.exploration_mult = exploration_mult
        self.current_idx = 0
        self.current_steps = self.k + K * self.exploration_mult
        self.K = K
        self.params_model_class = params_model_class
        self.params_model_class_config = params_model_class_config
        self.regularization_factor_mle = regularization_factor_mle
        self.mle_method = mle_method
        self.mle_recycle = mle_recycle
        self.mle_lr = mle_lr
        self.mle_steps = mle_steps
        self.last_grad = None
        self.params_model_bar = None

    def next_price_x(self, env, c):
        if self.current_idx < self.exploration_mult * self.K:
            p = (
                torch.rand(size=(1, env.model.K)) * (env.model.u - env.model.el)
                + env.model.el
            )
            x = env.model.x_set[np.random.randint(env.model.x_set.shape[0])]
        else:
            if self.current_idx == self.exploration_mult * self.K:
                (
                    self.params_model_bar,
                    self.last_grad,
                ) = MultipleMNLContextualModel.mle(
                    Cs=env.context_history[: env.t],
                    Is=env.choice_history[: env.t],
                    Ps=env.price_history[: env.t],
                    Xs=env.x_history[: env.t],
                    K=env.model.K,
                    context_size=env.model.context_size,
                    params_model_class=self.params_model_class,
                    method=self.mle_method,
                    recycle=self.mle_recycle,
                    steps=self.mle_steps * (self.exploration_mult * self.K + self.k),
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
            p, x = model_res.max_mean_demand_price_x(c)
        self.current_idx += 1
        if self.current_idx == self.current_steps:
            self.k += 1
            self.current_idx = 0
            self.current_steps = self.k + self.exploration_mult * self.K
        data = {}
        if self.last_grad is not None:
            data["last_grad"] = self.last_grad
        return (p, x), data
