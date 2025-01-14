import numpy as np
from scipy.optimize import fminbound


def bellman_operator(
    w,
    grid,
    beta,
    u,
    f_production,
    f_state_transition,
    Tw=None,
    compute_policy: bool = False,
):
    # === Apply linear interpolation to w === #
    w_func = lambda x: np.interp(x, grid, w)

    # == Initialize Tw if necessary == #
    if Tw is None:
        Tw = np.empty_like(w)

    if compute_policy:
        policy = np.empty_like(w)
        production_ = np.empty_like(w)
        control = np.empty_like(w)

    # == set Tw[i] = max_c { u(c) + β E w(f(y  - c) z)} == #
    for i, h in enumerate(grid):

        def objective(l, h=h):
            return -u(f_production(h, l)) - (beta * w_func(f_state_transition(h, l)))

        l_star = fminbound(objective, 0, 1)

        if compute_policy:
            policy[i] = f_state_transition(h, l_star)
            production_[i] = f_production(h, l_star)
            control[i] = l_star

        Tw[i] = -objective(l_star)

    if compute_policy:
        return Tw, policy, production_, control
    else:
        return Tw


def solve_optgrowth(
    initial_w, grid, beta, u, f_production, f_state_transition, tol=1e-4, max_iter=500
):

    w = initial_w  # Set initial condition
    error = tol + 1
    i = 0

    # == Create storage array for bellman_operator. Reduces  memory
    # allocation and speeds code up == #
    Tw = np.empty(len(grid))

    # Iterate to find solution
    while error > tol and i < max_iter:
        w_new = bellman_operator(
            w=w,
            grid=grid,
            beta=beta,
            u=u,
            f_production=f_production,
            f_state_transition=f_state_transition,
            Tw=Tw,
        )
        # error = #Your code goes here
        error = np.max(np.abs(w_new - w))
        w[:] = w_new
        i += 1
        (
            print("Iteration " + str(i) + "\n Error is " + str(error) + "\n")
            if i % 50 == 0 or error < tol
            else None
        )

    # Computes policy
    w, policy, production, control = bellman_operator(
        w=w,
        grid=grid,
        beta=beta,
        u=u,
        f_production=f_production,
        f_state_transition=f_state_transition,
        Tw=Tw,
        compute_policy=True,
    )

    return [w, policy, production, control]


# TODO add a function to check that the policy chosen is in interior of the feasible set each time ex-post
