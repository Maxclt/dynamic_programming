import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm


class HyperParameters:
    def save_hyperparameters(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class CES(HyperParameters):

    def __init__(
        self,
        alpha: float = 0.4,
        beta: float = 0.9,
        delta: float = 0.05,
        sigma: float = 0.9,
        h_0: float = 1,
    ):
        self.save_hyperparameters(
            alpha=alpha, beta=beta, delta=delta, sigma=sigma, h_0=h_0
        )

    def u(self, c):
        return c ** (1 - self.sigma) / (1 - self.sigma)

    def u_prime(self, c):
        return c ** (-self.sigma)

    def output(self, h, l):
        return h**self.alpha * l

    def motion(self, h, l):
        return (1 - self.delta) * h + 1 - l


class GridData(HyperParameters):
    """Generates grid data."""

    def __init__(self, max_T: int = 32, batch_size: int = 8):
        self.save_hyperparameters(max_T=max_T, batch_size=batch_size)
        self.time_range = torch.arange(0.0, self.max_T, 1.0).unsqueeze(dim=1)
        self.loader = self._create_dataloader()

    def _create_dataloader(self) -> DataLoader:
        """Creates a DataLoader for the grid."""
        dataset = DataLabel(self.time_range)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


class DataLabel(Dataset):
    """Dataset class for labeled data."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class NN(nn.Module, HyperParameters):

    def __init__(
        self,
        dim_hidden=128,
        layers=4,
        hidden_bias=True,
        alpha=0.4,
        beta=0.9,
        delta=0.05,
        sigma=0.9,
        h_0=1,
    ):
        super().__init__()
        self.save_hyperparameters(
            dim_hidden=dim_hidden, layers=layers, hidden_bias=hidden_bias
        )
        self.ces = CES(alpha, beta, delta, sigma, h_0)

        # Define the network layers
        self.q = self.build_network()

    def build_network(self):
        # torch.manual_seed(123)
        module = [nn.Linear(1, self.dim_hidden, bias=self.hidden_bias)]
        module.append(nn.Tanh())
        for _ in range(self.layers - 1):
            module.append(
                nn.Linear(self.dim_hidden, self.dim_hidden, bias=self.hidden_bias)
            )
            module.append(nn.Tanh())
        module.append(nn.Linear(self.dim_hidden, 2))
        return nn.Sequential(*module)

    def forward(self, x):
        out = self.q(x)

        first_coef = F.softplus(out[:, 0], beta=1.0)  # Softplus for h (human capital)
        second_coef = F.sigmoid(out[:, 1])  # Sigmoid for l (labor)

        # Combine the coefficients into a single output tensor
        return torch.stack((first_coef, second_coef), dim=1)

    def compute_loss(
        self, time: torch.Tensor, weight: np.ndarray = (1 / 3) * np.array([1, 1, 1])
    ):
        # Time variables
        time_zero = torch.zeros([1, 1])  # t = 0
        time_next = time + 1

        # Neural network outputs for t, t+1, and t=0
        h_t = self(time)[:, 0]
        l_t = self(time)[:, 1]
        c_t = self.ces.output(h_t, l_t)
        h_tp1 = self(time_next)[:, 0]
        l_tp1 = self(time_next)[:, 1]
        c_tp1 = self.ces.output(h_tp1, l_tp1)
        h_t0 = self(time_zero)[:, 0]

        # Residuals (based on CES equations)
        res_1 = h_tp1 - self.ces.motion(h_t, l_t)
        res_2 = self.ces.u_prime(
            c_t
        ) * h_t**self.ces.alpha - self.ces.beta * self.ces.u_prime(c_tp1) * (
            self.ces.alpha * h_tp1 ** (self.ces.alpha - 1) * l_tp1
            + (1 - self.ces.delta) * h_tp1**self.ces.alpha
        )
        res_3 = h_t0 - self.ces.h_0

        # Loss terms
        loss_1 = res_1.pow(2).mean()  # Motion constraint loss
        loss_2 = res_2.pow(2).mean()  # Euler equation loss
        loss_3 = res_3.pow(2).mean()  # Initial condition loss

        # Weighted total loss
        loss = weight[0] * loss_1 + weight[1] * loss_2 + weight[2] * loss_3
        return loss, loss_1, loss_2, loss_3


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
