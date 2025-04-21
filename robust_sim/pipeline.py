import json
import numpy as np
import pandas as pd

from .config    import load
from .log       import get_logger
from .data      import prepare_data
from .risk      import calc_cov, compute_cvar
from .implied   import implied_mu
from .posterior import calibrate_tau, bl_posterior
from .optim     import solve_qp

def run(cfg_path="config.yaml", out_csv="weights.csv", refresh_data=False):
    # 1. Load configuration
    cfg = load(cfg_path)
    log = get_logger(cfg["log_level"])

    # 2. Prepare data
    exc = prepare_data(cfg, refresh=refresh_data)

    # Asset ordering
    eqs = cfg["assets"]["equities"]
    bds = cfg["assets"]["bonds"]
    rvs = cfg["assets"]["real_assets"]
    assets = eqs + bds + rvs

    # 3. Statistics
    mu_hat = exc.mean() * 12
    cov, sigma, n_obs = calc_cov(exc)

    # 4. Equilibrium return
    mu_eq = implied_mu(cov, delta=3.0, vt_path=cfg["vt_weights"])

    # 5. Posterior
    if cfg["bayesian"]:
        tau = calibrate_tau(mu_eq, cov)
        mu_use = bl_posterior(mu_eq, mu_hat.values, cov, sigma, n_obs, tau)
    else:
        mu_use = mu_eq

    # 6. VT vector
    vt_map = json.load(open(cfg["vt_weights"]))
    vt_vec = np.array([vt_map[a] for a in assets])

    # 7. Floors and caps
    floors = {
        "equity_floor": cfg["equity_floor"],
        "bond_floor":   cfg["bond_floor"],
        "real_floor":   cfg["real_floor"],
        "indices": (
            [assets.index(a) for a in eqs],
            [assets.index(a) for a in bds],
            [assets.index(a) for a in rvs]
        )
    }
    caps = [cfg["asset_caps"][a] for a in assets]

    # 8. Optimize
    weights = solve_qp(
        mu_use, cov, vt_vec, cfg["risk_aversion"],
        cfg["te_limit"], floors, caps, log
    )

    # 9. Report CVaR and TE
    cvar = compute_cvar(exc, weights, cfg["cvar_threshold"])
    te   = np.sqrt((weights-vt_vec) @ cov @ (weights-vt_vec))
    log.info(f"Resulting CVaR{int(100*cfg['cvar_threshold'])}%: {cvar:.4f}")
    log.info(f"Tracking error (ann.): {te:.4%}")

    # 10. Save weights
    pd.DataFrame([weights], columns=assets).to_csv(out_csv, index=False)
    log.info("Weights saved to %s", out_csv)

    return weights
