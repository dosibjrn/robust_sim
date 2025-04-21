import numpy as np, scipy.optimize as opt

def solve_qp(mu, cov, vt, gamma, te, eq_floor, caps, logger):
    n = len(mu)
    def obj(w):
        return -(w@mu) + 0.5*gamma*(w@cov@w)

    cons = [
      {"type":"eq",   "fun": lambda w: w.sum()-1},
      {"type":"ineq", "fun": lambda w: w[:len(vt)-1].sum()-eq_floor}
    ]
    if te:
        cons.append({"type":"ineq",
                     "fun": lambda w, C=cov, v=vt, lim=te**2:
                               lim - (w-v) @ C @ (w-v)})

    bounds = [(0, caps[i]) for i in range(n)]
    res = opt.minimize(obj, vt.copy(), method="SLSQP",
                       bounds=bounds, constraints=cons)
    if not res.success:
        logger.warning("QP failed: %s", res.message)
    return res.x
