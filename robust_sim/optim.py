# robust_sim/optim.py

import numpy as np
import scipy.optimize as opt

# Try to import cvxpy for optional CVaR constraint
try:
    import cvxpy as cp
    USE_CVX = True
except ImportError:
    USE_CVX = False

def solve_qp(mu, cov, vt, gamma, te, floors, caps, log,
             cvar_limit=None, excess=None, alpha=None):
    """
    Solve the portfolio optimization problem.

    Parameters
    ----------
    mu : np.ndarray
        Expected returns vector.
    cov : np.ndarray
        Covariance matrix.
    vt : np.ndarray
        Benchmark (VT) weights.
    gamma : float
        Risk aversion parameter.
    te : float or None
        Tracking-error limit (annualized).
    floors : dict
        Floor constraints with keys "equity_floor", "bond_floor", "real_floor",
        and index lists "eq", "bd", "rv".
    caps : list or np.ndarray
        Per-asset upper bounds.
    log : logging.Logger
        Logger for warnings.
    cvar_limit : float or None
        If set and cvxpy is installed, enforce CVaR constraint.
    excess : pandas.DataFrame
        Historical excess returns (for CVaR).
    alpha : float
        CVaR confidence level (e.g. 0.95).

    Returns
    -------
    w : np.ndarray
        Optimized weights.
    """
    n = len(mu)

    # -- CVaR-constrained QP via cvxpy if requested and available --
    if cvar_limit is not None and USE_CVX and excess is not None:
        w = cp.Variable(n)
        v = cp.Variable()
        z = cp.Variable(excess.shape[0])

        # Objective: maximize mean-variance utility
        obj = cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, cov))

        # Constraints
        cons = [
            cp.sum(w) == 1,
            w >= 0,
            w <= caps,
            cp.quad_form(w - vt, cov) <= te**2,
            cp.sum(w[floors["eq"]])  >= floors["equity_floor"],
            cp.sum(w[floors["bd"]])  >= floors["bond_floor"],
            cp.sum(w[floors["rv"]])  >= floors["real_floor"],
            z >= v - excess.values @ w,
            z >= 0,
            v + (1/(1 - alpha)) * (cp.sum(z) / excess.shape[0]) <= cvar_limit
        ]

        # Solve with OSQP if available, else ECOS
        solver = cp.OSQP if "OSQP" in cp.installed_solvers() else cp.ECOS
        prob = cp.Problem(obj, cons)
        prob.solve(solver=solver)

        return w.value

    # -- Fallback: standard mean-variance via SciPy SLSQP (no CVaR) --
    def objective(w):
        return -(w @ mu) + 0.5 * gamma * (w @ cov @ w)

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: np.sum(w[floors["eq"]]) - floors["equity_floor"]},
        {"type": "ineq", "fun": lambda w: np.sum(w[floors["bd"]]) - floors["bond_floor"]},
        {"type": "ineq", "fun": lambda w: np.sum(w[floors["rv"]]) - floors["real_floor"]}
    ]

    if te:
        cons.append({
            "type": "ineq",
            "fun": lambda w: te**2 - (w - vt) @ cov @ (w - vt)
        })

    bounds = [(0, caps[i]) for i in range(n)]

    res = opt.minimize(objective, vt, method="SLSQP",
                       bounds=bounds, constraints=cons)

    if not res.success:
        log.warning("QP did not converge: %s", res.message)

    return res.x
