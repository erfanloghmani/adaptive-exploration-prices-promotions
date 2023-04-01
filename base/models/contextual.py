import sys

sys.path.append("../../")

import numpy as np
import torch

from MLEPARAMS import *
from base.optimizers import SGLD


class MultipleMNLContextualModel:
    def __init__(
        self,
        context_size,
        K,
        params_model,
        marginal_costs,
        negative_beta_option,
        x_set=None,
        el=0,
        u=25,
    ):
        """Contextual Multinomial Logistic Demand Model

        Args:
            context_size (int): Context vector dimensionality.
            K (int): Number of products.
            params_model (torch.nn.Module): The model for the alpha, beta, and gamma parameters.
            marginal_costs (torch.tensor): Marginal costs of each product.
            negative_beta_option (str): How to deal with the negative betas.
            x_set (torch.tensor, optional): Set of possible promotions, each row contains one promotion scenario on K products. Defaults to None.
            el (float, optional): Lower bound of prices. Defaults to 0.
            u (float, optional): Upper bound of prices. Defaults to 25.
        """
        self.K = K
        self.context_size = context_size
        self.params_model = params_model
        self.marginal_costs = marginal_costs
        self.x_set = x_set
        self.el = el
        self.u = u
        self.negative_beta_option = negative_beta_option

    def get_params(self, c):
        """Get alpha, beta, and gamma parameters for the given context.

        Args:
            c (torch.tensor): Context vector.

        Returns:
            tuple: Tuple of alpha, beta, and gamma parameters.
        """
        params = self.params_model(c)
        return (
            params[:, : self.K],
            params[:, self.K : 2 * self.K],
            params[:, 2 * self.K :],
        )

    @staticmethod
    def U_f_static(params_model, c, p, x, K):
        """Static version of the utility function for products at specific context, prices and promotions

        Args:
            params_model (torch.nn.Module): The model for the alpha, beta, and gamma parameters.
            c (torch.tensor): Context vector.
            p (torch.tensor): Prices
            x (torch.tensor): Promotions
            K (int): Number of products.
        Returns:
            torch.tensor: Utilities
        """
        params = params_model(c)
        alphas, betas, gammas = (
            params[:, :K],
            params[:, K : 2 * K],
            params[:, 2 * K :],
        )
        U = alphas - betas * p + gammas * x
        d = 1 if len(U.shape) == 1 else (U.shape[0], 1)
        return torch.cat([torch.zeros(d), U], dim=len(U.shape) - 1)

    def U_f(self, c, p, x):
        """Utility function for products at specific context, prices and promotions

        Args:
            c (torch.tensor): Context vector
            p (torch.tensor): Prices
            x (torch.tensor): Promotions
        Returns:
            torch.tensor: Utilities
        """
        alphas, betas, gammas = self.get_params(c)
        U = alphas - betas * p + gammas * x
        d = 1 if len(U.shape) == 1 else (U.shape[0], 1)
        return torch.cat([torch.zeros(d), U], dim=len(U.shape) - 1)

    def Q_f(self, c, p, x):
        """Mean demand function for products at specific context, prices and promotions

        Args:
            c (torch.tensor): Context vector
            p (torch.tensor): Prices
            x (torch.tensor): Promotions

        Returns:
            torch.tensor: Quantities (sum to 1)
        """
        if len(p.shape) == 1:
            return torch.softmax(self.U_f(c, p, x), 0)
        else:
            return torch.softmax(self.U_f(c, p, x), 1)

    def mean_demand(self, c, p, x):
        """Mean demand for products without outside option at specific context, prices and promotions

        Args:
            c (torch.tensor): Context vector
            p (torch.tensor): Prices
            x (torch.tensor): Promotions

        Returns:
            torch.tensor: Quantities without outside option
        """
        if len(p.shape) == 1:
            return self.Q_f(c, p, x)[1:]
        else:
            return self.Q_f(c, p, x)[:, 1:]

    def mean_profit(self, c, p, x):
        """Mean profit at specific context, prices and promotions

        Args:
            c (torch.tensor): Context vector
            p (torch.tensor): Prices
            x (torch.tensor): Promotions

        Returns:
            torch.tensor: Mean profit
        """
        demand = self.mean_demand(c, p, x)
        return demand.matmul((p - self.marginal_costs).T)

    def sample_choice(self, c, p, x):
        """Sample a choice at specific context, prices and promotions

        Args:
            c (torch.tensor): Context vector
            p (torch.tensor): Prices
            x (torch.tensor): Promotions

        Returns:
            int: User choice
        """
        q_f = self.Q_f(c, p, x)[0, :]
        return np.random.choice(np.arange(self.K + 1), p=q_f.detach().numpy())

    def max_mean_demand_price_for_fixed_x(self, c, x):
        """Price that Maximum mean demand happens at specific context and promotion setting

        Args:
            c (torch.tensor): Context vector
            x (torch.tensor): Promotion vector

        Returns:
            torch.tensor: Optimal price vector
        """
        alphas, betas, gammas = self.get_params(c)
        alphas_with_x = alphas + gammas * x

        def rhs(b_hat):
            try:
                res = (
                    1
                    / betas
                    * torch.exp(
                        -(1 + betas * b_hat + betas * self.marginal_costs)
                        + alphas_with_x
                    )
                )
                return res.sum()
            except:
                print("exception", (-(1 + betas * b_hat) + alphas_with_x, b_hat))

        def binary_search(start, end):
            if end - start < 1e-5:
                return start
            mid = (start + end) / 2
            if rhs(mid) - mid > 0:
                return binary_search(mid, end)
            else:
                return binary_search(start, mid)

        B = 1
        while True:
            if rhs(B) - B < 0:
                break
            B *= 1.5
        B0 = binary_search(0, B)
        if self.negative_beta_option == "zero":
            return torch.where(
                betas > 0,
                (1 / betas + B0 + self.marginal_costs),
                torch.zeros((1, self.K)),
            )
        elif self.negative_beta_option == "max":
            return torch.where(
                betas > 0,
                (1 / betas + B0 + self.marginal_costs),
                torch.ones((1, self.K)) * self.u,
            )
        else:
            raise Exception(
                f"Negative beta option {self.negative_beta_option} not found"
            )

    def max_mean_demand_price_x(self, c):
        """Price that maximum mean demand happens at specific context.

        Args:
            c (torch.tensor): Context vector

        Returns:
            torch.tensor: Optimal price vector
        """
        assert c.shape[0] == 1
        max_rev, max_p, max_x = -1, None, None
        for i, x in enumerate(self.x_set):
            best_price_for_x = self.max_mean_demand_price_for_fixed_x(c, x)
            mean_demand = self.mean_demand(c, best_price_for_x, x)
            rev = mean_demand.matmul(best_price_for_x.T)[0, 0]
            if rev > max_rev:
                max_rev = rev
                max_p = best_price_for_x
                max_x = x
        return max_p, max_x

    def max_mean_demand(self, c):
        """Maximum mean demand at specific context.

        Args:
            c (torch.tensor): Context vector

        Returns:
            torch.tensor: Maximum mean demand
        """
        p, x = self.max_mean_demand_price_x(c)
        return self.mean_demand(c, p, x)

    @classmethod
    def mle(
        cls,
        Cs,
        Is,
        Ps,
        Xs,
        K,
        context_size,
        params_model_class,
        params_model_class_config,
        recycle,
        lr,
        steps=None,
        params_model_s_old=None,
        method="GD",
        regularization=None,
    ):
        """Find the optimal parameter model from the historical data using the MLE method

        Args:
            Cs (torch.tensor): Context history
            Is (torch.tensor): Decision history
            Ps (torch.tensor): Price history
            Xs (torch.tensor): Promotion history
            K (int): Number of products
            context_size (int): Context vector dimensionality.
            params_model_class (Class): Parameter model class, e.g., ParameterModelLinear
            params_model_class_config (dict): Parameters passed to init method of Parameter Model Class
            recycle (bool, optional): Whether to recycle the parameters, if True use the _old values to start MLE. Defaults to False.
            lr (float, optional): Learning rate.
            steps (int, optional): Number of optimization steps, if None use the default. Defaults to None.
            params_model_s_old (torch.nn.Module, optional): Previous parameter model to start with. Defaults to None.
            method (str, optional): MLE method to use. Defaults to "GD".
            regularization (float, optional): Regularization parameter. Defaults to None.

        Returns:
            tuple: Tuple of parameter model and the norm of the gradient at final step.
        """
        assert method in ["GD", "BFGS"]
        if not recycle:
            params_model_s_old = None
        if method == "GD":
            return MultipleMNLContextualModel._mle(
                Cs=Cs,
                Is=Is,
                Ps=Ps,
                Xs=Xs,
                K=K,
                context_size=context_size,
                params_model_class=params_model_class,
                params_model_s_old=params_model_s_old,
                lr=lr,
                regularization=regularization,
                params_model_class_config=params_model_class_config,
                steps=steps,
            )
        elif method == "BFGS":
            return MultipleMNLContextualModel._mle_lbfgs(
                Cs=Cs,
                Is=Is,
                Ps=Ps,
                Xs=Xs,
                K=K,
                context_size=context_size,
                params_model_class=params_model_class,
                params_model_s_old=params_model_s_old,
                regularization=regularization,
                params_model_class_config=params_model_class_config,
            )

    @classmethod
    def _mle(
        cls,
        Cs,
        Is,
        Ps,
        Xs,
        K,
        context_size,
        params_model_class,
        params_model_class_config,
        params_model_s_old,
        steps=None,
        lr=None,
        regularization=None,
    ):
        if params_model_s_old is None:
            params_model_s = params_model_class(
                context_size, K, **params_model_class_config
            )
        else:
            params_model_s = params_model_s_old
            for p in params_model_s.parameters():
                p.requires_grad = True
        if lr is None:
            lr = MLEGD_LR
        if steps is None:
            steps = MLEGD_TOTAL_ITERS
        optimizer = torch.optim.SGD(params_model_s.parameters(), lr=lr)
        loss = torch.nn.CrossEntropyLoss(reduction="sum")
        cnt = 0
        while cnt < steps:
            cnt += 1

            params = params_model_s(Cs)
            alphas, betas, gammas = (
                params[:, :K],
                params[:, K : 2 * K],
                params[:, 2 * K :],
            )
            U_f = torch.zeros((Cs.shape[0], K + 1))
            U_f[:, 1:] = alphas - betas * Ps + gammas * Xs
            ll = loss(U_f, Is)
            if regularization:
                for name, W in params_model_s.named_parameters():
                    if "weight" in name:
                        ll += regularization * (W.norm(2) ** 2)
            ll = ll / len(Is)
            ll.backward()
            grad_norm = 0
            for name, W in params_model_s.named_parameters():
                if "weight" in name:
                    grad_norm += W.grad.norm(2) ** 2
            optimizer.step()
            optimizer.zero_grad()

        for p in params_model_s.parameters():
            p.requires_grad = False
        return params_model_s, grad_norm.item()

    @classmethod
    def _mle_lbfgs(
        cls,
        Cs,
        Is,
        Ps,
        Xs,
        K,
        context_size,
        params_model_class,
        params_model_class_config,
        params_model_s_old,
        regularization=None,
    ):
        if params_model_s_old is None:
            params_model_s = params_model_class(
                context_size, K, **params_model_class_config
            )
        else:
            params_model_s = params_model_s_old
            for p in params_model_s.parameters():
                p.requires_grad = True
        optimizer = torch.optim.LBFGS(
            params_model_s.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=MLEBFGS_OPT_ITERS,
        )

        cnt = 0
        grad_norms = []

        while cnt < MLEBFGS_TOTAL_ITERS:
            cnt += 1

            def closure():
                loss = torch.nn.CrossEntropyLoss(reduction="sum")
                model_s = MultipleMNLContextualModel(
                    context_size,
                    K,
                    params_model_s,
                    marginal_costs=torch.zeros(n),
                    negative_beta_option="max",
                )
                optimizer.zero_grad()
                U_f = model_s.U_f(Cs, Ps, Xs)
                ll = loss(U_f, Is)
                if regularization:
                    for name, W in params_model_s.named_parameters():
                        if "weight" in name:
                            ll += regularization * (W.norm(2) ** 2)
                ll = ll / len(Is)
                ll.backward()
                grad_norm = 0
                for name, W in params_model_s.named_parameters():
                    if "weight" in name:
                        grad_norm += W.grad.norm(2) ** 2
                grad_norms.append(grad_norm.item())
                return ll

            optimizer.step(closure)

        for p in params_model_s.parameters():
            p.requires_grad = False
        return params_model_s, grad_norms[-1]

    @classmethod
    def langevin_step(
        cls,
        Cs,
        Is,
        Ps,
        Xs,
        K,
        context_size,
        params_model_class,
        N_t,
        eta,
        psi_inverse,
        params_model_s_old=None,
        params_model_class_config={},
        regularization=None,
    ):
        """Perform a Langevin Dynamics step on the historical data

        Args:
            Cs (torch.tensor): Context history
            Is (torch.tensor): Decision history
            Ps (torch.tensor): Price history
            Xs (torch.tensor): Promotion history
            K (int): Number of products
            context_size (int): Dimensionality of the context vector
            params_model_class (Class): Parameter model class, e.g., ParameterModelLinear
            params_model_class_config (dict): Parameters passed to init method of Parameter Model Class
            N_t (int): Number of langevin inner loop steps
            eta (float): Eta parameter
            psi_inverse (float): psi_inverse parameter
            params_model_s_old (torch.nn.Module, optional): Previous parameter model to start with. Defaults to None.
            regularization (float, optional): Regularization parameter. Defaults to None.

        Returns:
            tuple: Tuple of parameter model and the norm of the gradient at final step.
        """
        if params_model_s_old is not None:
            params_model_s = params_model_s_old
            for p in params_model_s.parameters():
                p.requires_grad = True
        else:
            params_model_s = params_model_class(
                context_size, K, **params_model_class_config
            )
        optimizer = SGLD(params_model_s.parameters(), lr=eta, beta=psi_inverse)
        loss = torch.nn.CrossEntropyLoss(reduction="sum")
        cnt = 0
        while cnt < N_t:
            cnt += 1
            model_s = MultipleMNLContextualModel(
                context_size,
                K,
                params_model_s,
                marginal_costs=torch.zeros(K),
                negative_beta_option="max",
            )
            U_f = model_s.U_f(Cs, Ps, Xs)
            ll = loss(U_f, Is)
            if regularization:
                for name, W in params_model_s.named_parameters():
                    if "weight" in name:
                        ll += regularization * (W.norm(2) ** 2)
            ll.backward()
            grad_norm = 0
            for name, W in params_model_s.named_parameters():
                if "weight" in name:
                    grad_norm += (W.grad / len(Is)).norm(2)
            optimizer.step()
            optimizer.zero_grad()
        for p in params_model_s.parameters():
            p.requires_grad = False
        return params_model_s, grad_norm.item()
