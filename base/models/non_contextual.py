import sys

sys.path.append("../../")

import numpy as np
import torch

from MLEPARAMS import *


class MultipleMNLModel:
    def __init__(self, alphas, betas, gammas, marginal_costs, x_set=None, el=0, u=25):
        """NonContextual Multinomial Logistic Demand Model

        Args:

            alphas (torch.tensor): Alpha parameters
            betas (torch.tensor): Beta parameters
            gammas (torch.tensor): Gamma parameters
            marginal_costs (torch.tensor): Marginal costs of each product.
            x_set (torch.tensor, optional): Set of possible promotions, each row contains one promotion scenario on K products. Defaults to None.
            el (float, optional): Lower bound of prices. Defaults to 0.
            u (float, optional): Upper bound of prices. Defaults to 25.
        """
        self.K = alphas.shape[0]
        self.alphas = alphas
        self.betas = betas
        self.gammas = gammas
        self.marginal_costs = marginal_costs
        self.x_set = x_set
        self.el = el
        self.u = u

    def U_f(self, p, x):
        """Utility function for products at specific prices and promotions

        Args:
            p (torch.tensor): Prices
            x (torch.tensor): Promotions

        Returns:
            torch.tensor: Utilities
        """
        U = self.alphas - self.betas * p + self.gammas * x
        d = 1 if len(p.shape) == 1 else (p.shape[0], 1)
        return torch.cat([torch.zeros(d), U], dim=len(p.shape) - 1)

    def Q_f(self, p, x):
        """Mean demand function for products at specific prices and promotions

        Args:
            p (torch.tensor): Prices
            x (torch.tensor): Promotions

        Returns:
            torch.tensor: Quantities (sum to 1)
        """
        if len(p.shape) == 1:
            return torch.softmax(self.U_f(p, x), 0)
        else:
            return torch.softmax(self.U_f(p, x), 1)

    def mean_demand(self, p, x):
        """Mean demand for products without outside option at specific prices and promotions

        Args:
            p (torch.tensor): Prices
            x (torch.tensor): Promotions

        Returns:
            torch.tensor: Quantities without outside option
        """
        if len(p.shape) == 1:
            return self.Q_f(p, x)[1:]
        else:
            return self.Q_f(p, x)[:, 1:]

    def mean_profit(self, p, x):
        """Mean profit at specific prices and promotions

        Args:
            p (torch.tensor): Prices
            x (torch.tensor): Promotions

        Returns:
            torch.tensor: Mean profit
        """
        demand = self.mean_demand(p, x)
        return demand.dot(p - self.marginal_costs)

    def sample_choice(self, p, x):
        """Sample a choice at specific prices and promotions

        Args:
            p (torch.tensor): Prices
            x (torch.tensor): Promotions

        Returns:
            int: User choice
        """
        q_f = self.Q_f(p, x)
        return np.random.choice(np.arange(self.K + 1), p=q_f.detach().numpy())

    def max_mean_demand_price_for_fixed_x(self, x):
        """Price that Maximum mean demand happens at specific promotion setting

        Args:
            x (torch.tensor): Promotion vector

        Returns:
            torch.tensor: Optimal price vector
        """
        alphas_with_x = self.alphas + self.gammas * x

        def rhs(b_hat):
            try:
                res = (
                    1
                    / self.betas
                    * torch.exp(
                        -(1 + self.betas * b_hat + self.betas * self.marginal_costs)
                        + alphas_with_x
                    )
                )
                return res.sum()
            except:
                print("exception", (-(1 + self.betas * b_hat) + alphas_with_x, b_hat))

        def binary_search(start, end):
            if end - start < 1e-10:
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
        return torch.maximum(
            torch.tensor([0]), 1 / self.betas + B0 + self.marginal_costs
        )

    def max_mean_demand_price_x(self):
        """Price that maximum mean demand happens

        Returns:
            torch.tensor: Optimal price vector
        """
        max_rev, max_p, max_x = -1, None, None
        for i, x in enumerate(self.x_set):
            best_price_for_x = self.max_mean_demand_price_for_fixed_x(x)
            mean_demand = self.mean_demand(best_price_for_x, x)
            rev = mean_demand.dot(best_price_for_x)
            if rev > max_rev:
                max_rev = rev
                max_p = best_price_for_x
                max_x = x
        return max_p, max_x

    def max_mean_demand(self):
        """Maximum mean demand of the model

        Returns:
            torch.tensor: Maximum mean demand
        """
        p, x = self.max_mean_demand_price_x()
        return self.mean_demand(p, x)

    @classmethod
    def mle(
        cls,
        Is,
        Ps,
        Xs,
        K,
        alphas_s_old=None,
        betas_s_old=None,
        gammas_s_old=None,
        steps=None,
        method="GD",
        recycle=False,
        lr=None,
        regularization=None,
    ):
        """Find the optimal parameters from the historical data using the MLE method

        Args:
            Is (torch.tensor): Decision history
            Ps (torch.tensor): Price history
            Xs (torch.tensor): Promotion history
            K (int): Number of products
            alphas_s_old (torch.tensor, optional): Previous alphas to start with. Defaults to None.
            betas_s_old (torch.tensor, optional): Previous betas to start with. Defaults to None.
            gammas_s_old (torch.tensor, optional): Previous gammas to start with. Defaults to None.
            steps (int, optional): Number of optimization steps, if None use the default. Defaults to None.
            method (str, optional): MLE method to use. Defaults to "GD".
            recycle (bool, optional): Whether to recycle the parameters, if True use the _old values to start MLE. Defaults to False.
            lr (float, optional): Learning rate. Defaults to None.
            regularization (float, optional): Regularization parameter. Defaults to None.

        Returns:
            (tuple): Tuple containing:
                alphas (torch.tensor): Alpha parameters
                betas (torch.tensor): Beta parameters
                gammas (torch.tensor): Gamma parameters
                grad_norm (float): Gradient norm of the last optimization step
        """
        assert method in ["GD", "BFGS"]
        if not recycle:
            alphas_s_old = None
            betas_s_old = None
            gammas_s_old = None
        if method == "GD":
            return MultipleMNLModel._mle(
                Is=Is,
                Ps=Ps,
                Xs=Xs,
                K=K,
                lr=lr,
                alphas_s_old=alphas_s_old,
                betas_s_old=betas_s_old,
                gammas_s_old=gammas_s_old,
                regularization=regularization,
                steps=steps,
            )
        elif method == "BFGS":
            return MultipleMNLModel._mle_lbfgs(
                Is=Is,
                Ps=Ps,
                Xs=Xs,
                K=K,
                alphas_s_old=alphas_s_old,
                betas_s_old=betas_s_old,
                gammas_s_old=gammas_s_old,
                regularization=regularization,
            )

    @classmethod
    def _mle(
        cls,
        Is,
        Ps,
        Xs,
        K,
        alphas_s_old=None,
        betas_s_old=None,
        gammas_s_old=None,
        regularization=None,
        lr=None,
        steps=None,
    ):
        if (
            alphas_s_old is not None
            and betas_s_old is not None
            and gammas_s_old is not None
        ):
            alphas_s = torch.autograd.Variable(
                alphas_s_old.detach().clone(), requires_grad=True
            )
            betas_s = torch.autograd.Variable(
                betas_s_old.detach().clone(), requires_grad=True
            )
            gammas_s = torch.autograd.Variable(
                gammas_s_old.detach().clone(), requires_grad=True
            )
        else:
            alphas_s = torch.autograd.Variable(torch.rand(K), requires_grad=True)
            betas_s = torch.autograd.Variable(torch.rand(K), requires_grad=True)
            gammas_s = torch.autograd.Variable(torch.rand(K), requires_grad=True)
        if lr is None:
            lr = MLEGD_LR
        if steps is None:
            steps = MLEGD_TOTAL_ITERS
        optimizer = torch.optim.Adam([alphas_s, betas_s, gammas_s], lr=lr)
        loss = torch.nn.CrossEntropyLoss(reduction="sum")
        cnt = 0
        grad_norm = 10
        while cnt < steps:
            cnt += 1
            model_s = MultipleMNLModel(
                alphas_s, betas_s, gammas_s, marginal_costs=torch.zeros(K)
            )
            U_f = model_s.U_f(Ps, Xs)
            ll = loss(U_f, Is)
            if regularization:
                ll += regularization * (
                    torch.norm(alphas_s) ** 2
                    + torch.norm(betas_s) ** 2
                    + torch.norm(gammas_s) ** 2
                )
            ll = ll / len(Is)
            ll.backward()
            grad_norm = torch.sqrt(
                torch.norm(alphas_s.grad) ** 2
                + torch.norm(betas_s.grad) ** 2
                + torch.norm(gammas_s.grad) ** 2
            )
            grad_norm = grad_norm.item()
            optimizer.step()
            optimizer.zero_grad()
        return (
            alphas_s.detach().clone(),
            betas_s.detach().clone(),
            gammas_s.detach().clone(),
            grad_norm,
        )

    @classmethod
    def _mle_lbfgs(
        cls,
        Is,
        Ps,
        Xs,
        K,
        alphas_s_old=None,
        betas_s_old=None,
        gammas_s_old=None,
        regularization=None,
    ):
        if (
            alphas_s_old is not None
            and betas_s_old is not None
            and gammas_s_old is not None
        ):
            alphas_s = torch.autograd.Variable(
                alphas_s_old.detach().clone(), requires_grad=True
            )
            betas_s = torch.autograd.Variable(
                betas_s_old.detach().clone(), requires_grad=True
            )
            gammas_s = torch.autograd.Variable(
                gammas_s_old.detach().clone(), requires_grad=True
            )
        else:
            alphas_s = torch.autograd.Variable(torch.rand(K), requires_grad=True)
            betas_s = torch.autograd.Variable(torch.rand(K), requires_grad=True)
            gammas_s = torch.autograd.Variable(torch.rand(K), requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [alphas_s, betas_s, gammas_s],
            line_search_fn="strong_wolfe",
            max_iter=MLEBFGS_OPT_ITERS,
        )

        cnt = 0
        grad_norms = []

        while cnt < MLEBFGS_TOTAL_ITERS:
            cnt += 1

            def closure():
                loss = torch.nn.CrossEntropyLoss(reduction="sum")
                model_s = MultipleMNLModel(
                    alphas_s, betas_s, gammas_s, marginal_costs=torch.zeros(K)
                )
                optimizer.zero_grad()
                ll = loss(model_s.U_f(Ps, Xs), Is)
                if regularization:
                    ll += regularization * (
                        torch.norm(alphas_s) ** 2
                        + torch.norm(betas_s) ** 2
                        + torch.norm(gammas_s) ** 2
                    )
                ll = ll / len(Is)
                ll.backward()
                grad_norm = torch.sqrt(
                    torch.norm(alphas_s.grad) ** 2
                    + torch.norm(betas_s.grad) ** 2
                    + torch.norm(gammas_s.grad) ** 2
                )
                grad_norms.append(grad_norm.item())
                return ll

            optimizer.step(closure)

        return (
            alphas_s.detach().clone(),
            betas_s.detach().clone(),
            gammas_s.detach().clone(),
            grad_norms[-1],
        )

    @classmethod
    def langevin_step(
        cls,
        Is,
        Ps,
        Xs,
        K,
        N_t,
        eta,
        psi_inverse,
        alphas_s_old=None,
        betas_s_old=None,
        gammas_s_old=None,
        regularization=None,
    ):
        """Perform a Langevin Dynamics step on the historical data

        Args:
            Is (torch.tensor): Decision history
            Ps (torch.tensor): Price history
            Xs (torch.tensor): Promotion history
            K (int): Number of products
            N_t (int): Number of langevin inner loop steps
            eta (float): Eta parameter
            psi_inverse (float): psi_inverse parameter
            alphas_s_old (torch.tensor, optional): Previous alphas to start with. Defaults to None.
            betas_s_old (torch.tensor, optional): Previous betas to start with. Defaults to None.
            gammas_s_old (torch.tensor, optional): Previous gammas to start with. Defaults to None.
            regularization (float, optional): Regularization parameter. Defaults to None.

        Returns:
            (tuple): Tuple containing:
                alphas (torch.tensor): Alpha parameters
                betas (torch.tensor): Beta parameters
                gammas (torch.tensor): Gamma parameters
                grad_norm (float): Gradient norm of the last optimization step
        """
        if (
            alphas_s_old is not None
            and betas_s_old is not None
            and gammas_s_old is not None
        ):
            alphas_s = torch.autograd.Variable(
                alphas_s_old.detach().clone(), requires_grad=True
            )
            betas_s = torch.autograd.Variable(
                betas_s_old.detach().clone(), requires_grad=True
            )
            gammas_s = torch.autograd.Variable(
                gammas_s_old.detach().clone(), requires_grad=True
            )
        else:
            alphas_s = torch.autograd.Variable(torch.ones(K), requires_grad=True)
            betas_s = torch.autograd.Variable(torch.ones(K), requires_grad=True)
            gammas_s = torch.autograd.Variable(torch.ones(K), requires_grad=True)
        loss = torch.nn.CrossEntropyLoss(reduction="sum")
        cnt = 0
        while cnt < N_t:
            cnt += 1
            model_s = MultipleMNLModel(
                alphas_s, betas_s, gammas_s, marginal_costs=torch.zeros(K)
            )
            U_f = model_s.U_f(Ps, Xs)
            ll = loss(U_f, Is)
            if regularization:
                ll += regularization * (
                    torch.norm(alphas_s) ** 2
                    + torch.norm(betas_s) ** 2
                    + torch.norm(gammas_s) ** 2
                )
            ll.backward()
            with torch.no_grad():
                grad_norm = torch.sqrt(
                    torch.norm(alphas_s.grad) ** 2
                    + torch.norm(betas_s.grad) ** 2
                    + torch.norm(gammas_s.grad) ** 2
                ) / len(Is)
                grad_norm = grad_norm.item()
                alphas_s -= eta * alphas_s.grad + np.sqrt(
                    2 * eta / psi_inverse
                ) * torch.randn(K)
                alphas_s.grad.zero_()
                betas_s -= eta * betas_s.grad + np.sqrt(
                    2 * eta / psi_inverse
                ) * torch.randn(K)
                betas_s.grad.zero_()

                gammas_s -= eta * gammas_s.grad + np.sqrt(
                    2 * eta / psi_inverse
                ) * torch.randn(K)
                gammas_s.grad.zero_()

        return (
            alphas_s.detach().clone(),
            betas_s.detach().clone(),
            gammas_s.detach().clone(),
            grad_norm,
        )
