import numpy as np, scipy.optimize as opt

def solve_qp(mu, cov, vt_vec, gamma, te, eq_floor, caps, logger):
    n = len(mu)
    def objective(w):   # negative MV utility
        return -(w @ mu) + 0.5 * gamma * (w @ cov @ w)

    cons = [{"type":"eq",   "fun": lambda w: w.sum()-1},
            {"type":"ineq", "fun": lambda w: w[0:4].sum() - eq_floor}]  # first 4 = equities & BTC
    if te is not None:
        cons.append({"type":"ineq",
                     "fun": lambda w, C=cov, v=vt_vec, lim=te**2:
                                      lim - (w-v) @ C @ (w-v)})

    bounds = [(0, caps[i]) for i in range(n)]
    res = opt.minimize(objective, vt_vec.copy(),
                       method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        logger.warning("SLSQP failed: %s", res.message)
    return res.x
