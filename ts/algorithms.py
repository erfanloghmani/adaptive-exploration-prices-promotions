import numpy as np
import torch
from base.models.non_contextual import MultipleMNLModel
from base.models.contextual import MultipleMNLContextualModel
from base.parameter_models import ParameterModelLinear


class ThompsonSamplingLaplace:
    def __init__(
        self,
        tau,
        exploration_rate,
        K,
        batch_size,
        regularization_factor_H,
        regularization_factor_mle,
        fix_model,
        mle_method,
        mle_recycle,
        mle_lr,
    ):
        """Thompson Sampling for NonContextual Prices and Promotions Experiments

        Args:
            tau (int): Number of initial exploration rounds
            exploration_rate (float): Multipicative factor on the normal distribution variance
            K (int): Number of products
            batch_size (int): Batch size
            regularization_factor_H (float): Regularization factor on the Hessian matrix
            regularization_factor_mle (float): Regulatization factor for MLE
            fix_model (boolean): Whether to fix the model during the experimentation phase or resample every round
            mle_method (string): MLE method
            mle_recycle (boolean): Whether to recycle the parameters from previous step for MLE
            mle_lr (float): Learning rate of MLE
        """
        self.K = K
        self.tau = tau
        self.exploration_rate = exploration_rate
        self.H = regularization_factor_H * torch.eye(3 * K)
        self.batch_size = batch_size

        self.regularization_factor_H = regularization_factor_H
        self.regularization_factor_mle = regularization_factor_mle
        self.mle_method = mle_method
        self.mle_recycle = mle_recycle
        self.mle_lr = mle_lr
        self.fix_model = fix_model
        self.last_grad = None

    def next_price_x(self, env, c=None):
        """What price to play at the current state of the environment

        Args:
            env (Environment): Environment (could be contextual or not)
            c (torch.array, optional): Placeholder for context which is not used for this non-contextual method

        Returns:
            tuple: A tuple of optimal action (price, promotion), and additional data
        """

        def hessian_for_sample(model, p, x):
            q = model.mean_demand(p, x).unsqueeze(1)
            mult = torch.cat([torch.ones(self.K), -p, x]).unsqueeze(0)
            H_A = torch.diag(-q[:, 0]) + q.matmul(q.T)
            H_A_repeat = H_A.repeat(3, 3)
            return -(mult * H_A_repeat) * mult.T

        if env.t < self.tau:
            p = torch.rand(self.K) * (env.model.u - env.model.el) + env.model.el
            x = env.model.x_set[np.random.randint(env.model.x_set.shape[0])]
            data = {}
            self.alpha_bar, self.beta_bar, self.gamma_bar = None, None, None
        else:
            if (env.t - self.tau) % self.batch_size == 0:
                (
                    self.alpha_bar,
                    self.beta_bar,
                    self.gamma_bar,
                    self.last_grad,
                ) = MultipleMNLModel.mle(
                    env.choice_history[: env.t],
                    env.price_history[: env.t],
                    env.x_history[: env.t],
                    env.model.K,
                    alphas_s_old=self.alpha_bar,
                    betas_s_old=self.beta_bar,
                    gammas_s_old=self.gamma_bar,
                    method=self.mle_method,
                    recycle=self.mle_recycle,
                    lr=self.mle_lr,
                    regularization=self.regularization_factor_mle,
                )
                self.theta_bar = torch.cat(
                    [self.alpha_bar, self.beta_bar, self.gamma_bar]
                )
                self.model_bar = MultipleMNLModel(
                    self.alpha_bar,
                    self.beta_bar,
                    self.gamma_bar,
                    marginal_costs=env.model.marginal_costs,
                    x_set=env.model.x_set,
                )
                if env.t == self.tau:
                    for i, p in enumerate(env.price_history):
                        self.H += hessian_for_sample(
                            self.model_bar, p, env.x_history[i]
                        )
                self.H_fixed = self.H.clone()

            if not self.fix_model or (env.t - self.tau) % self.batch_size == 0:
                theta_res_dist = (
                    torch.distributions.multivariate_normal.MultivariateNormal(
                        self.theta_bar,
                        precision_matrix=((self.exploration_rate**-2) * self.H_fixed),
                        validate_args=False,
                    )
                )
                theta_res = theta_res_dist.sample()
                self.alpha_res = theta_res[: self.K]
                self.beta_res = theta_res[self.K : 2 * self.K]
                self.gamma_res = theta_res[2 * self.K :]

            model_res = MultipleMNLModel(
                self.alpha_res,
                self.beta_res,
                self.gamma_res,
                marginal_costs=env.model.marginal_costs,
                x_set=env.model.x_set,
            )
            p, x = model_res.max_mean_demand_price_x()
            self.H += hessian_for_sample(self.model_bar, p, x)
            data = {
                "alpha_bar": self.theta_bar[: self.K],
                "beta_bar": self.theta_bar[self.K : 2 * self.K],
                "gamma_bar": self.theta_bar[2 * self.K :],
            }
            if self.last_grad is not None:
                data["last_grad"] = self.last_grad
        return (p, x), data


class ThompsonSamplingLangevin:
    def __init__(
        self,
        tau,
        K,
        batch_size,
        N_t,
        eta_scale,
        decay,
        regularization_factor_mle,
        fix_model,
        psi_inverse,
    ):
        """Langevin Sampling for NonContextual Prices and Promotions Experiments

        Args:
            tau (int): Number of initial exploration rounds
            K (int): Number of products
            batch_size (int): Batch size
            N_t (int): Number of MCMC steps
            eta_scale (float): Step size
            decay (bool): Whether to decay the step size inversely with time
            regularization_factor_mle (float): Regulatization factor for MLE
            fix_model (boolean): Whether to fix the model during the experimentation phase or resample every round
            psi_inverse (float): Multipicative factor on the noise variance
        """

        self.K = K
        self.tau = tau
        self.batch_size = batch_size
        self.regularization_factor_mle = regularization_factor_mle
        self.fix_model = fix_model
        self.N_t = N_t
        self.eta_scale = eta_scale
        self.psi_inverse = psi_inverse
        self.decay = decay
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
            p = torch.rand(self.K) * (env.model.u - env.model.el) + env.model.el
            x = env.model.x_set[np.random.randint(env.model.x_set.shape[0])]
            data = {}
            self.alpha_bar, self.beta_bar, self.gamma_bar = None, None, None
        else:
            if not self.fix_model or (env.t - self.tau) % self.batch_size == 0:
                last_time = env.t - ((env.t - self.tau) % self.batch_size)
                if self.fix_model:
                    N_t = self.N_t * self.batch_size
                else:
                    N_t = self.N_t
                result = MultipleMNLModel.langevin_step(
                    env.choice_history[:last_time],
                    env.price_history[:last_time],
                    env.x_history[:last_time],
                    K=self.K,
                    N_t=N_t,
                    eta=self.eta_scale / last_time if self.decay else self.eta_scale,
                    psi_inverse=self.psi_inverse,
                    alphas_s_old=self.alpha_bar,
                    betas_s_old=self.beta_bar,
                    gammas_s_old=self.gamma_bar,
                    regularization=self.regularization_factor_mle,
                )
                self.alpha_bar, self.beta_bar, self.gamma_bar, self.last_grad = result

                self.model_bar = MultipleMNLModel(
                    self.alpha_bar,
                    self.beta_bar,
                    self.gamma_bar,
                    marginal_costs=env.model.marginal_costs,
                    x_set=env.model.x_set,
                )
            p, x = self.model_bar.max_mean_demand_price_x()
            data = {
                "alpha_bar": self.alpha_bar,
                "beta_bar": self.beta_bar,
                "gamma_bar": self.gamma_bar,
            }
            if self.last_grad is not None:
                data["last_grad"] = self.last_grad
        return (p, x), data


class ContextualThompsonSamplingLangevin:
    def __init__(
        self,
        tau,
        K,
        decay,
        params_model_class,
        batch_size,
        N_t,
        N_t_explore,
        eta_scale,
        psi_inverse,
        regularization_factor_mle,
        batch,
        params_model_class_config={},
    ):
        """Contextual Langevin Sampling for Prices and Promotions Experiments

        Args:
            tau (int): Number of initial exploration rounds
            K (int): Number of products
            decay (bool): Whether to decay the step size inversely with time
            params_model_class (Class): Parameter model class, e.g., ParameterModelLinear
            batch_size (int): Batch size
            N_t (int): Number of MCMC steps
            N_t_explore (int): Number of MCMC steps in the batch steps
            eta_scale (float): Step size
            psi_inverse (float): Multipicative factor on the noise variance
            regularization_factor_mle (float): Regulatization factor for MLE
            batch (bool): Whether to use batching or not
            params_model_class_config (dict, optional): Parameters passed to init method of Parameter Model Class. Defaults to {}.
        """
        self.K = K
        if params_model_class is None:
            self.params_model_class = ParameterModelLinear
        else:
            self.params_model_class = params_model_class
        self.params_model_class_config = params_model_class_config
        self.tau = tau
        self.batch_size = batch_size
        self.regularization_factor_mle = regularization_factor_mle
        self.N_t = N_t
        self.N_t_explore = N_t_explore
        self.eta_scale = eta_scale
        self.decay = decay
        self.psi_inverse = psi_inverse
        self.params_model_bar = None
        self.batch = batch

    def next_price_x(self, env, c):
        """What price to play at the current state of the environment

        Args:
            env (Environment): Contextual environment
            c (torch.array): Context vector

        Returns:
            tuple: A tuple of optimal action (price, promotion), and additional data
        """

        def find_model(env, N_t):
            last_time = env.t

            (
                params_model_bar,
                last_grad,
            ) = MultipleMNLContextualModel.langevin_step(
                Cs=env.context_history[:last_time],
                Is=env.choice_history[:last_time],
                Ps=env.price_history[:last_time],
                Xs=env.x_history[:last_time],
                K=env.model.K,
                context_size=env.model.context_size,
                params_model_class=self.params_model_class,
                N_t=N_t,
                eta=self.eta_scale / last_time if self.decay else self.eta_scale,
                psi_inverse=self.psi_inverse,
                params_model_s_old=self.params_model_bar,
                regularization=self.regularization_factor_mle,
                params_model_class_config=self.params_model_class_config,
            )

            model_bar = MultipleMNLContextualModel(
                env.model.context_size,
                env.model.K,
                params_model_bar,
                marginal_costs=env.model.marginal_costs,
                x_set=env.model.x_set,
                negative_beta_option=env.model.negative_beta_option,
                el=env.model.el,
                u=env.model.u,
            )
            return params_model_bar, last_grad, model_bar

        if env.t < self.tau:
            p = (
                torch.rand(size=(1, self.K)) * (env.model.u - env.model.el)
                + env.model.el
            )
            x = env.model.x_set[np.random.randint(env.model.x_set.shape[0])]
            data = {}
            self.params_model_bar = None
        else:
            if self.batch:
                if (env.t - self.tau) % self.batch_size == 0:
                    self.params_model_bar, self.last_grad, self.model_bar = find_model(
                        env, self.N_t
                    )
                else:
                    self.params_model_bar, self.last_grad, self.model_bar = find_model(
                        env, self.N_t_explore
                    )
            else:
                self.params_model_bar, self.last_grad, self.model_bar = find_model(
                    env, self.N_t
                )
            p, x = self.model_bar.max_mean_demand_price_x(c)
            data = {"last_grad": self.last_grad}

        return (p, x), data
