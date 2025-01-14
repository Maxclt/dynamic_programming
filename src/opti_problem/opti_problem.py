class OptiProblem:

    def __init__(
        self,
        alpha: float = 0.9,
        beta: float = 0.9,
        delta: float = 0.05,
        sigma: float = 0.9,
    ):
        """Initiate the hyperparameters of the problem

        Args:
            alpha (float, optional): Elasticity of human capital in the production function. Defaults to 0.9.
            beta (float, optional): Coefficient of relative risk aversion. Defaults to 0.9.
            delta (float, optional): Depreciation rate of human capital. Defaults to 0.05.
            sigma (float, optional): Discount factor. Defaults to 0.9.
        """
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.sigma = sigma

    def u(self, c: float) -> float:
        """Define the agent's CRRA utility function

        Args:
            c (float): A consumption level

        Returns:
            float: the agent's CRRA utility for a consumption level
        """
        return (c ** (1 - self.sigma)) / (1 - self.sigma)

    def f_production(self, H: float, L: float) -> float:
        """_summary_

        Args:
            H (float): _description_
            L (float): _description_

        Returns:
            float: _description_
        """
        return (H**self.alpha) * L

    def f_state_transion(self, H: float, L: float) -> float:
        """

        Args:
            H (float): _description_
            L (float): _description_

        Returns:
            float: _description_
        """
        return (1 - self.delta) * H + (1 - L)
