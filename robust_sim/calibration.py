# robust_sim/calibration.py

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

def compute_forward_1yr_returns(prices):
    fwd = prices.shift(-12) / prices - 1
    return fwd.dropna()

def calibrate_shr_tau(cfg, taus, lookback_months):
    log = get_logger(cfg["log_level"])
    exc    = prepare_data(cfg)
    # Build price series for realized returns
    eq = pd.read_csv(cfg["equity_csv"], parse_dates=["Date"], index_col="Date")
    bd = pd.read_csv(cfg["bond_csv"],   parse_dates=["Date"], index_col="Date")
    wd = pd.read_csv(cfg["wood_csv"])
    wd["Date"] = pd.to_datetime(wd["Year"].astype(str), format="%Y")
    wd = wd.set_index("Date")[["Mänty (€)","Kuusi (€)","Koivu (€)"]]
    wd_m = wd.resample("ME").interpolate().mean(axis=1).to_frame("FinWood")
    prices = pd.concat([eq, bd, wd_m], axis=1).resample("ME").last().dropna()
    realized = compute_forward_1yr_returns(prices)

    errors = {τ: [] for τ in taus}
    for i in range(lookback_months, len(exc) - 12):
        train = exc.iloc[i - lookback_months : i]
        date  = exc.index[i]
        if train.shape[0] < lookback_months or date not in realized.index:
            continue

        mu_hat = train.mean() * 12
        cov, sigma, n = calc_cov(train)
        mu_eq = implied_mu(cov)

        for τ in taus:
            mu_post = bl_posterior(mu_eq, mu_hat.values, cov, sigma, n, τ)
            err = np.linalg.norm(mu_post - realized.loc[date].values)
            errors[τ].append(err)

    # Average RMSE per τ, skipping those with no samples
    avg_err = {τ: np.mean(v) for τ, v in errors.items() if len(v) > 0}
    if not avg_err:
        log.error("Calibration of shr_tau failed: no data points. Using default %s", cfg["shr_tau"])
        return cfg["shr_tau"]

    best = min(avg_err, key=avg_err.get)
    log.info("Calibrated shr_tau=%.4f (avg RMSEs=%s)", best, avg_err)
    return best

def calibrate_te_limit(cfg, tes, lookback_months):
    log = get_logger(cfg["log_level"])
    exc    = prepare_data(cfg)
    vt_map = json.load(open(cfg["vt_weights"]))
    assets = list(cfg["assets"]["equities"]) + list(cfg["assets"]["bonds"]) + list(cfg["assets"]["real_assets"])
    vt_vec = np.array([vt_map[a] for a in assets])
    floors = {
        "equity_floor": cfg["equity_floor"],
        "bond_floor":   cfg["bond_floor"],
        "real_floor":   cfg["real_floor"],
        "eq": [assets.index(a) for a in cfg["assets"]["equities"]],
        "bd": [assets.index(a) for a in cfg["assets"]["bonds"]],
        "rv": [assets.index(a) for a in cfg["assets"]["real_assets"]],
    }
    caps = [cfg["asset_caps"][a] for a in assets]

    sharpe_out = {}
    for te in tes:
        sharps = []
        for i in range(lookback_months, len(exc) - 12):
            train = exc.iloc[i - lookback_months : i]
            test  = exc.iloc[i : i + 12]
            date  = exc.index[i]
            if train.shape[0] < lookback_months or len(test) < 12:
                continue

            mu_hat = train.mean() * 12
            cov, sigma, n = calc_cov(train)
            mu_eq = implied_mu(cov)
            mu_use = bl_posterior(mu_eq, mu_hat.values, cov, sigma, n, cfg["shr_tau"]) if cfg["bayesian"] else mu_eq

            w = solve_qp(mu_use, cov, vt_vec,
                         cfg["risk_aversion"], te,
                         floors, caps, log)
            ret_fwd = test.dot(w)
            if ret_fwd.std() > 0:
                sharps.append(ret_fwd.mean() / ret_fwd.std())

        if sharps:
            sharpe_out[te] = np.mean(sharps)

    if not sharpe_out:
        log.error("Calibration of te_limit failed: no valid backtest windows. Using default %s", cfg["te_limit"])
        return cfg["te_limit"]

    best_te = max(sharpe_out, key=sharpe_out.get)
    log.info("Calibrated te_limit=%.4f (Sharpe=%s)", best_te, sharpe_out)
    return best_te


# at top of calibration.py
from .pipeline import run  # for reuse, if needed

def calibrate_gamma(cfg, gammas, lookback_months):
    """
    Grid‑search over gamma values. For each gamma:
      • In each rolling window, optimize portfolio with that gamma,
        using current cfg["te_limit"], cfg["shr_tau"], etc.
      • Compute the 12‑month forward realized utility:
            U = E[r_fwd] - 0.5*gamma*Var(r_fwd)
      • Average U across windows.
    Return the gamma with highest mean U.
    """
    log = get_logger(cfg["log_level"])
    exc    = prepare_data(cfg)
    vt_map = json.load(open(cfg["vt_weights"]))
    assets = list(cfg["assets"]["equities"]) + \
             list(cfg["assets"]["bonds"])    + \
             list(cfg["assets"]["real_assets"])
    vt_vec = np.array([vt_map[a] for a in assets])
    floors = {
        "equity_floor": cfg["equity_floor"],
        "bond_floor":   cfg["bond_floor"],
        "real_floor":   cfg["real_floor"],
        "eq": [assets.index(a) for a in cfg["assets"]["equities"]],
        "bd": [assets.index(a) for a in cfg["assets"]["bonds"]],
        "rv": [assets.index(a) for a in cfg["assets"]["real_assets"]],
    }
    caps = [cfg["asset_caps"][a] for a in assets]

    # Precompute forward returns
    fwd_ret = compute_forward_1yr_returns(
        pd.concat([
            pd.read_csv(cfg["equity_csv"], parse_dates=["Date"], index_col="Date"),
            pd.read_csv(cfg["bond_csv"],   parse_dates=["Date"], index_col="Date"),
            pd.read_csv(cfg["wood_csv"])
              .assign(Date=lambda df: pd.to_datetime(df["Year"].astype(str),format="%Y"))
              .set_index("Date")[["Mänty (€)","Kuusi (€)","Koivu (€)"]]
              .resample("ME")
              .interpolate()
              .mean(axis=1)
              .to_frame("FinWood")
        ],axis=1)
        .resample("ME").last()
    )

    avg_util = {}
    for γ in gammas:
        utils = []
        for i in range(lookback_months, len(exc) - 12):
            train = exc.iloc[i - lookback_months : i]
            test  = fwd_ret.iloc[i]    # a Series of 1‑yr forward returns
            if train.shape[0] < lookback_months: 
                continue

            mu_hat = train.mean() * 12
            cov, sigma, n = calc_cov(train)
            mu_eq  = implied_mu(cov)
            mu_use = bl_posterior(mu_eq, mu_hat.values, cov, sigma, n, cfg["shr_tau"]) \
                     if cfg["bayesian"] else mu_eq

            # Optimize with this gamma:
            w = solve_qp(mu_use, cov, vt_vec,
                         γ, cfg["te_limit"],
                         floors, caps, log)

            # Compute realized 12‑month forward utility:
            r = test.dot(w)
            var = test.dot(w)**2  # scalar in this one‑period case
            U = r - 0.5 * γ * var
            utils.append(U)

        if utils:
            avg_util[γ] = np.mean(utils)

    if not avg_util:
        log.error("No valid windows for gamma calibration; using default %s", cfg["risk_aversion"])
        return cfg["risk_aversion"]

    best = max(avg_util, key=avg_util.get)
    log.info("Calibrated risk_aversion=%.4f (avgU=%s)", best, avg_util)
    return best
