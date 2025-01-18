import matplotlib.pyplot as plt
import numpy as np

from abc import ABC, abstractmethod


class CES(ABC):

    def __init__(
        self,
        init_w: np.ndarray,
        grid: np.ndarray,
        alpha: float = 0.4,
        beta: float = 0.9,
        delta: float = 0.05,
        sigma: float = 0.9,
    ):
        """Initiate the hyperparameters of the problem

        Args:
            alpha (float, optional): Elasticity of human capital in the output function. Defaults to 0.4.
            beta (float, optional): Coefficient of relative risk aversion. Defaults to 0.9.
            delta (float, optional): Depreciation rate of human capital. Defaults to 0.05.
            sigma (float, optional): Discount factor. Defaults to 0.9.
        """
        self.init_w, self.grid = init_w, grid
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.sigma = sigma

        self.results = {"Value": None, "Policy": None, "Output": None, "Control": None}

    def u(self, c: float) -> float:
        """Define the agent's CRRA utility function

        Args:
            c (float): A consumption level

        Returns:
            float: the agent's CRRA utility for a consumption level
        """
        return (c ** (1 - self.sigma)) / (1 - self.sigma)

    def f_output(self, H: float, L: float) -> float:
        """_summary_

        Args:
            H (float): _description_
            L (float): _description_

        Returns:
            float: _description_
        """
        return (H**self.alpha) * L

    def f_state_transition(self, H: float, L: float) -> float:
        """

        Args:
            H (float): _description_
            L (float): _description_

        Returns:
            float: _description_
        """
        return (1 - self.delta) * H + (1 - L)

    # H_new = (1-self.delta)*H_old + (1-L) then 1-L = H_new - (1-self.delta)*H_old

    def inv_state_transition(self, H_new: float, H_old: float) -> float:
        return 1 - H_new + (1 - self.delta) * H_old

    @abstractmethod
    def solve_optgrowth(tol=1e-4, max_iter=500):
        pass

    def get_results(self):
        return self.results

    def plot_results(
        self, var: str, method: str = "recursive", control_bound: tuple = (0, 1)
    ):

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_ylim(min(self.results[var]), max(self.results[var]))
        ax.plot(
            self.grid,
            self.results[var],
            lw=2,
            alpha=0.6,
            label="approximate value function",
        )

        if var in {"Policy", "Output", "State"}:
            ax.plot(self.grid, self.grid, lw=2, alpha=0.6, label="45 degrees line")

        if var in {"Policy"}:
            above_upper = next(
                (
                    self.grid[i]
                    for i in range(len(self.results[var]))
                    if self.results[var][i] > control_bound[1]
                ),
                None,
            )
            below_lower = next(
                (
                    self.grid[i]
                    for i in range(len(self.results[var]))
                    if self.results[var][i] < control_bound[0]
                ),
                None,
            )
            if above_upper:
                plt.axvline(
                    x=above_upper,
                    color="red",
                    linestyle="--",
                    label=f"Domain frontier reached above x = {above_upper:.2f}",
                )
            if below_lower:
                plt.axvline(
                    x=below_lower,
                    color="blue",
                    linestyle="--",
                    label=f"Domain frontier reached below x = {below_lower:.2f}",
                )

        ax.set_xlabel("Human Capital")
        ax.set_ylabel(var)
        ax.legend(loc="lower right")
        plt.show()
