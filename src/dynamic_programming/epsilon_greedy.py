import numpy as np

from joblib import Parallel, delayed
from scipy.optimize import fminbound, minimize_scalar
from tqdm import tqdm

from src.problem.opti_problem import CES

# TODO take the course example with imrpove policy and modify to include exploration.


class EpsilonGreedy(CES):

    def __init__(self, *args, epsilon: float = 1e-4, **kwargs):
        super().__init__(*args, **kwargs)

        self.epsilon = epsilon

    # ==============================================================================
    # Optimal Growth: Policy Iteration
    # ==============================================================================

    def evaluation_policy(self, w, pol):
        # === Apply linear interpolation to w === #
        w_func = lambda x: np.interp(x, self.grid, w)

        Tw = np.empty_like(w)

        # == set Tw[i] = max_c { u(c) + β E w(f(y  - c))} == #
        for i, h in enumerate(self.grid):
            Tw[i] = self.u(self.f_output(h, pol[i])) + self.beta * w_func(
                self.f_state_transition(h, pol[i])
            )

        return Tw

    def improve_greedy_policy(self, w, Tw=None):
        # === Apply linear interpolation to w === #
        w_func = lambda x: np.interp(x, self.grid, w)

        # == Initialize Tw if necessary == #
        if Tw is None:
            Tw = np.empty_like(w)

        Tpolicy = np.empty_like(w)

        # == set Tw[i] = max_c { u(c) + β E w(f(y  - c))} == #
        for i, h in enumerate(self.grid):

            def objective(l):
                if self.f_output(h, l) <= 0:
                    return np.inf
                else:
                    return -self.u(self.f_output(h, l)) - self.beta * w_func(
                        self.f_state_transition(h, l)
                    )

            if np.random.rand() > self.epsilon / len(self.grid):
                l_new = minimize_scalar(objective, bounds=(0, 1), method="bounded").x
            else:
                l_new = np.random.rand()

            Tpolicy[i] = l_new

            Tw[i] = -objective(l_new)

        return Tw, Tpolicy

    def solve_optgrowth_policy(self, w, pol, tol=1e-4, max_iter=500):

        error = tol + 1
        i = 0

        # Iterate to find solution
        while error > tol and i < max_iter:
            w_new = self.evaluation_policy(w, pol)
            error = np.max(np.abs(w_new - w))
            w = w_new
            i += 1

        print(i)

        return [w]

    def solve_optgrowth(self, tol=0.0001, max_iter=500):

        w = self.init_w
        error = tol + 1
        i = 0
        self.w_history = []
        self.pol_history = []

        # == Create storage array for bellman_operator. Reduces  memory
        # allocation and speeds code up == #
        Tw = np.empty_like(self.init_w)

        # First iter
        w, pol = self.improve_greedy_policy(w, Tw)

        with tqdm(total=max_iter, desc="Greedy Solver", unit="iter") as pbar:
            while error > tol and i < max_iter:
                w_approx = self.solve_optgrowth_policy(w, pol, tol=1e-4, max_iter=500)[
                    0
                ]
                w_new, pol_new = self.improve_greedy_policy(w_approx, Tw=None)
                error = np.max(np.abs(pol_new - pol))
                pol = pol_new
                w = w_new
                self.w_history.append(w_approx.copy())
                self.pol_history.append(pol.copy())  # Append a copy of w to the list

                if i % 2 == 0 or error < tol:
                    pbar.set_postfix({"Error": error})

                pbar.update(1)

                i += 1

        self.results["Value"], self.results["Policy"] = w_new, pol_new
