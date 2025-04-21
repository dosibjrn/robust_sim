import numpy as np
import scipy.optimize as opt

def solve_qp(mu, cov, vt, gamma,
             te_limit, floors, caps, logger):
    """
    Quadratic program:
      maximize  w' mu - 0.5 gamma w' cov w
      subject to sum(w)=1, w>=0,
                 TE^2=(w-vt)'cov(w-vt) <= te_limit^2,
                 group floors, per-asset caps.
    """
    n = len(mu)

    # Objective: negative utility
    def obj(w):
        return -(w @ mu) + 0.5 * gamma * (w @ cov @ w)

    # Constraints
    cons = [{"type":"eq", "fun": lambda w: w.sum() - 1}]
    # Tracking error
    if te_limit:
        cons.append({
            "type":"ineq",
            "fun": lambda w: te_limit**2 - (w - vt) @ cov @ (w - vt)
        })
    # Group floors
    eq_idx, bd_idx, rv_idx = floors["indices"]
    cons.append({"type":"ineq","fun": lambda w: w[eq_idx].sum() - floors["equity_floor"]})
    cons.append({"type":"ineq","fun": lambda w: w[bd_idx].sum() - floors["bond_floor"]})
    cons.append({"type":"ineq","fun": lambda w: w[rv_idx].sum() - floors["real_floor"]})

    # Bounds
    bounds = [(0, caps[i]) for i in range(n)]

    # Initial guess: vt
    w0 = vt.copy()

    res = opt.minimize(obj, w0, method="SLSQP",
                       bounds=bounds, constraints=cons)
    if not res.success:
        logger.warning("QP solver did not converge: %s", res.message)
    return res.x
