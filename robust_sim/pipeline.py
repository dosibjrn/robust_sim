import json
import numpy as np
import pandas as pd

from .config    import load
from .log       import get_logger
from .data      import prepare_data
from .risk      import calc_cov
from .implied   import implied_mu
from .posterior import bl_posterior
from .optim     import solve_qp

def run(cfg_path="config.yaml", out_csv="weights.csv", refresh_data=False):
    # 1. Load configuration and logger
    cfg = load(cfg_path)
    log = get_logger(cfg["log_level"])

    # 2. Determine asset order: equities + bonds + wood
    eq_names   = list(cfg["assets"]["equities"].keys())
    bond_names = list(cfg["assets"]["bonds"].keys())
    assets     = eq_names + bond_names + ["FinWood"]

    # 3. Load and prepare excess returns
    exc = prepare_data(cfg, refresh=refresh_data)[assets]

    # 4. Compute sample stats: annualized mean (mu_samp), covariance, sigma vector, n_obs
    mu_samp, cov, sigma, n_obs = exc.mean() * 12, *calc_cov(exc)

    # 5. Load VT weights and build VT vector in same asset order
    vt_map = json.load(open("data/vt_weights.json"))
    vt_vec = np.array([vt_map[a] for a in assets])

    # 6. Market‑implied equilibrium return
    mu_eq = implied_mu(cov)

    # 7. Optionally blend with historical via Black–Litterman
    if cfg["bayesian"]:
        mu_use = bl_posterior(mu_eq, mu_samp.values, cov, sigma, n_obs, cfg["shr_tau"])
    else:
        mu_use = mu_eq

    # 8. Solve the constrained MV QP
    w = solve_qp(
        mu_use,
        cov,
        vt_vec,
        cfg["risk_aversion"],
        cfg["te_limit"],
        cfg["equity_floor"],
        2 * vt_vec,    # per-asset cap = 2×VT weight
        log
    )

    # 9. Save the resulting weights
    pd.DataFrame([w], columns=assets).to_csv(out_csv, index=False)
    log.info("Weights saved to %s", out_csv)

    return w
