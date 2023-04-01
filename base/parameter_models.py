import torch


def get_parameter_model_class(c_str):
    if c_str == "ParameterModelNN":
        return ParameterModelNN
    if c_str == "ParameterModelNNNotShared":
        return ParameterModelNNNotShared
    if c_str == "ParameterModelLinear":
        return ParameterModelLinear
    if c_str == "NearestCenter":
        return NearestCenter
    else:
        raise Exception(f"Unknown Parameter Class {c_str}")


class ParameterModelLinear(torch.nn.Module):
    def __init__(self, context_size, K):
        super().__init__()
        self.context_size = context_size
        self.K = K
        self.p = torch.nn.Sequential(
            torch.nn.Linear(context_size, K * 3, bias=False),
        )
        self.p[0].weight.data = torch.ones((K * 3, context_size))

    def forward(self, x):
        x = self.p(x)
        return torch.cat(
            [x[:, : self.K], x[:, self.K : self.K * 2], x[:, 2 * self.K :]], dim=1
        )


class ParameterModelNN(torch.nn.Module):
    def __init__(self, context_size, K, hidden_size=3):
        super().__init__()
        self.context_size = context_size
        self.K = K
        self.hidden_size = hidden_size
        self.p = torch.nn.Sequential(
            torch.nn.Linear(context_size, self.hidden_size, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_size, K * 3, bias=True),
        )

    def forward(self, x):
        x = self.p(x)
        return torch.cat(
            [x[:, : self.K], x[:, self.K : self.K * 2], x[:, 2 * self.K :]], dim=1
        )

    def get_extra_state(self):
        return {"hidden_size": self.hidden_size}

    def set_extra_state(self, state):
        self.hidden_size = state["hidden_size"]


class ParameterModelNNNotShared(torch.nn.Module):
    def __init__(self, context_size, K, hidden_size=3):
        super().__init__()
        self.context_size = context_size
        self.K = K
        self.hidden_size = hidden_size
        self.p_alpha = torch.nn.Sequential(
            torch.nn.Linear(context_size, self.hidden_size, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_size, K, bias=True),
        )
        self.p_beta = torch.nn.Sequential(
            torch.nn.Linear(context_size, self.hidden_size, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_size, K, bias=True),
        )
        self.p_gamma = torch.nn.Sequential(
            torch.nn.Linear(context_size, self.hidden_size, bias=True),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_size, K, bias=True),
        )

    def forward(self, x):
        alpha_hat = self.p_alpha(x)
        beta_hat = self.p_beta(x)
        gamma_hat = self.p_gamma(x)
        return torch.cat([alpha_hat, beta_hat, gamma_hat], dim=1)

    def get_extra_state(self):
        return {"hidden_size": self.hidden_size}

    def set_extra_state(self, state):
        self.hidden_size = state["hidden_size"]


class NearestCenter(torch.nn.Module):
    def __init__(self, context_size, K, n_groups=3):
        super().__init__()
        self.context_size = context_size
        self.K = K
        self.n_groups = n_groups
        self.centers = torch.nn.Parameter(torch.zeros((n_groups, context_size)))
        self.center_values = torch.nn.Parameter(torch.zeros((3 * K, n_groups)))

    def forward(self, x):
        nearest = torch.linalg.vector_norm(
            self.centers.repeat(x.shape[0], 1, 1) - x.unsqueeze(1), dim=2
        ).argmin(dim=1)
        return self.center_values[:, nearest].T

    def get_extra_state(self):
        return {"n_groups": self.n_groups}

    def set_extra_state(self, state):
        self.n_groups = state["n_groups"]
