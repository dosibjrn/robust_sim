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

    If a CVaR limit is specified and cvxpy with a conic solver (ECOS/SCS/CVXOPT) is available,
    enforce CVaR constraint. Otherwise, fall back to SciPy SLSQP without CVaR.
    """
    n = len(mu)

    # Attempt CVaR-constrained QP via cvxpy if requested
    if cvar_limit is not None and USE_CVX and excess is not None:
        # Check for available conic solvers
        conic_solvers = [s for s in cp.installed_solvers() if s in ("ECOS", "SCS", "CVXOPT")]
        if not conic_solvers:
            log.warning("No conic solver available (need ECOS, SCS or CVXOPT) for CVaR constraint; falling back to SciPy.")
        else:
            w = cp.Variable(n)
            v = cp.Variable()
            z = cp.Variable(excess.shape[0])

            obj = cp.Maximize(mu @ w - 0.5 * gamma * cp.quad_form(w, cov))

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

            # Pick a conic solver
            solver = cp.ECOS if "ECOS" in conic_solvers else cp.SCS
            try:
                prob = cp.Problem(obj, cons)
                prob.solve(solver=solver)
                return w.value
            except Exception as e:
                log.warning("CVaR QP with %s failed: %s; falling back to SciPy.", solver, e)

    # Fallback: standard mean-variance via SciPy SLSQP (no CVaR)
    def objective(w):
        return -(w @ mu) + 0.5 * gamma * (w @ cov @ w)

    cons = [
        {"type": "eq",   "fun": lambda w: np.sum(w) - 1},
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
