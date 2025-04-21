import json, numpy as np, pandas as pd
from .config     import load
from .log        import get_logger
from .data       import prepare_data
from .risk       import calc_cov
from .implied    import implied_mu
from .posterior  import bl_posterior
from .optim      import solve_qp

def run(cfg_path="config.yaml", out_csv="weights.csv", refresh_data=False):
    cfg  = load(cfg_path)
    log  = get_logger(cfg["log_level"])

    # Asset order: equities + bonds + FinWood
    eq_names   = list(cfg["assets"]["equities"].keys())
    bond_names = list(cfg["assets"]["bonds"].keys())
    all_assets = eq_names + bond_names + ["FinWood"]

    exc = prepare_data(cfg, refresh=refresh_data)[all_assets]
    mu_s, cov, sigma, n_obs = exc.mean()*12, *calc_cov(exc)

    vt = json.load(open("data/vt_weights.json"))
    vt_vec = np.array([vt[a] for a in all_assets])

    mu_eq  = implied_mu(cov)
    mu_use = bl_posterior(mu_eq, mu_s.values, cov, sigma, n_obs, cfg["shr_tau"]) \
             if cfg["bayesian"] else mu_eq

    weights = solve_qp(
        mu_use, cov, vt_vec,
        cfg["risk_aversion"],
        cfg["te_limit"],
        cfg["equity_floor"],
        caps = 2 * vt_vec,
        logger = log
    )

    pd.DataFrame([weights], columns=all_assets).to_csv(out_csv, index=False)
    log.info("Weights saved to %s", out_csv)
    return weights
