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
    cfg = load(cfg_path)
    log = get_logger(cfg["log_level"])

    # 1. Prepare data
    exc = prepare_data(cfg, refresh=refresh_data)

    eqs = cfg["assets"]["equities"]
    bds = cfg["assets"]["bonds"]
    rvs = cfg["assets"]["real_assets"]
    assets = eqs + bds + rvs

    # 2. Stats
    mu_hat = exc.mean()*12
    cov, sigma, n_obs = calc_cov(exc)

    # 3. Equilibrium
    mu_eq = implied_mu(cov, vt_path=cfg["vt_weights"])

    # 4. Posterior
    if cfg["bayesian"]:
        tau = calibrate_tau(mu_eq, cov)
        mu_use = bl_posterior(mu_eq, mu_hat.values, cov, sigma, n_obs, tau)
    else:
        mu_use = mu_eq

    # 5. VT vector
    vt_map = json.load(open(cfg["vt_weights"]))
    vt_vec = np.array([vt_map[a] for a in assets])

    # 6. Floors & caps
    floors = {
      "equity_floor":cfg["equity_floor"],
      "bond_floor":  cfg["bond_floor"],
      "real_floor":  cfg["real_floor"],
      "eq":[assets.index(a) for a in eqs],
      "bd":[assets.index(a) for a in bds],
      "rv":[assets.index(a) for a in rvs]
    }
    caps = [cfg["asset_caps"][a] for a in assets]

    # 7. Optimize
    weights = solve_qp(
        mu_use, cov, vt_vec, cfg["risk_aversion"],
        cfg["te_limit"], floors, caps, log,
        cvar_limit=cfg.get("cvar_limit"),
        excess=exc, alpha=cfg["cvar_threshold"]
    )

    # 8. Report
    te   = np.sqrt((weights-vt_vec) @ cov @ (weights-vt_vec))
    cvar = compute_cvar(exc, weights, cfg["cvar_threshold"])
    log.info(f"Tracking Error: {te:.4%}")
    log.info(f"CVaR{int(100*cfg['cvar_threshold'])}%: {cvar:.4%}")

    # 9. Save
    pd.DataFrame([weights], columns=assets).to_csv(out_csv, index=False)
    log.info("Weights saved to %s", out_csv)
    return weights
