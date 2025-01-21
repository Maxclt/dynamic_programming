import numpy as np

from joblib import Parallel, delayed
from scipy.optimize import fminbound
from tqdm import tqdm

from src.problem.opti_problem import CES


class BellmanOperator(CES):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def objective(self, l: float, h: float, w) -> float:
        """This method computes the value of the objective function for the
        given labor and human capital levels.

        Args:
            l (float): The current labor supply.
            h (float): The current human capital.

        Returns:
            float: The objective function value.
        """
        w_func = lambda x: np.interp(x, self.grid, w)

        return -self.u(self.f_output(h, l)) - (
            self.beta * w_func(self.f_state_transition(h, l))
        )

    def minimize_objective(self, h: float, w) -> tuple:
        """Compute the optimal labor supply given the current human capital.

        This function computes the optimal labor supply, next period human capital,
        output, and utility given the human capital using the  operator.

        Args:
            h (float): Current human capital.

        Returns:
            tuple: A tuple containing the optimal labor supply, next period human
            capital, output, and utility.
        """

        l_star = fminbound(self.objective, 0, 1, args=(h, w))

        return (
            l_star,
            self.f_state_transition(h, l_star),
            self.f_output(h, l_star),
            -self.objective(l_star, h, w),
        )

    def bellman_operator(
        self, w, Tw: np.ndarray, compute_policy: bool = False
    ) -> tuple:

        # Parallel processing of grid points
        results = Parallel(n_jobs=-1)(
            delayed(self.minimize_objective)(h, w) for h in self.grid
        )

        # Collect results into arrays
        if compute_policy:
            policy = np.empty_like(self.init_w)
            output_ = np.empty_like(self.init_w)
            states = np.empty_like(self.init_w)

        for i, (l_star, state, output, value) in enumerate(results):
            if compute_policy:
                policy[i] = l_star
                output_[i] = output
                states[i] = state
            Tw[i] = value

        return (Tw, policy, output_, states) if compute_policy else Tw

    def solve_optgrowth(self, tol: float = 1e-4, max_iter: int = 500) -> tuple:
        """Solve the optimal growth problem using value function iteration.

        This method iteratively applies the Bellman operator until convergence to
        find the optimal value function and policy.

        Args:
            tol (float, optional): The tolerance for convergence. Defaults to 1e-4.
            max_iter (int, optional): The maximum number of iterations. Defaults to 500.

        Returns:
            tuple: A tuple containing the optimal value function, policy, output,
            and control.
        """

        w = self.init_w.copy()
        error = tol + 1
        i = 0

        # == Create storage array for bellman_operator. Reduces  memory
        # allocation and speeds code up == #
        Tw = np.empty_like(self.init_w)

        # Iterate to find solution
        with tqdm(total=max_iter, desc="Recursive Solver", unit="iter") as pbar:
            while error > tol and i < max_iter:
                w_new = self.bellman_operator(w, Tw)
                error = np.max(np.abs(w_new - w))
                w[:] = w_new
                i += 1

                if i % 50 == 0 or error < tol:
                    pbar.set_postfix({"Error": error})

                pbar.update(1)

        # Computes policy
        (
            self.results["Value"],
            self.results["Policy"],
            self.results["Output"],
            self.results["Next State"],
        ) = self.bellman_operator(w, Tw, compute_policy=True)


# TODO add a function to check that the policy chosen is in interior of the feasible set each time ex-post
