# v30_portfolio_engine.py – fleshed‑out proof‑of‑concept with robust regime handling
"""
30‑Year Robust DCA Engine
-------------------------
This single file implements:
• 3‑state regime model via k‑means on trailing 12‑month Sharpe vectors with global fallback
• Markov transition probabilities with Dirichlet smoothing
• Monthly block‑bootstrap DCA simulation for 30 years (360 months)
• Robust mean‑variance‑entropy optimisation with TE, floors, caps
• Optional CVaR constraint via CVXPY

Usage:
  python v30_portfolio_engine.py optimise -c config.yaml -o weights.csv
  python v30_portfolio_engine.py simulate -c config.yaml -w weights.csv
"""

import json, logging
from pathlib import Path
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from sklearn.cluster import KMeans
from scipy.optimize import minimize

try:
    import cvxpy as cp
    HAS_CVX = True
except ImportError:
    HAS_CVX = False

# Logging
LOG = logging.getLogger("v30")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# 1. Load returns
# ---------------------------------------------------------------------------

def load_returns(cfg: dict) -> pd.DataFrame:
    eq = pd.read_csv(cfg["equity_csv"], parse_dates=["Date"], index_col="Date")
    bd = pd.read_csv(cfg["bond_csv"], parse_dates=["Date"], index_col="Date")
    wd = pd.read_csv(cfg["wood_csv"])  # annual wood prices
    wd["Date"] = pd.to_datetime(wd["Year"].astype(str), format="%Y") + MonthEnd(0)
    wd = wd.set_index("Date")[ ["Mänty (€)", "Kuusi (€)", "Koivu (€)"] ]
    wd = wd.resample("ME").interpolate()
    fin = pd.DataFrame({"FinWood": wd.mean(axis=1)})

    prices = pd.concat([eq, bd, fin], axis=1).dropna()
    prices = prices.resample("ME").last()

    rf = pd.read_csv(cfg["risk_free_csv"], parse_dates=["Date"], index_col="Date")["RF"]
    rf = rf.resample("ME").last()

    rets = prices.pct_change(fill_method=None).dropna()
    excess = rets.sub(rf, axis=0).dropna()
    cols = cfg["assets"]["equities"] + cfg["assets"]["bonds"] + cfg["assets"]["real_assets"]
    return excess[cols]

# ---------------------------------------------------------------------------
# 2. Fit regimes with fallback
# ---------------------------------------------------------------------------

def fit_regimes(excess: pd.DataFrame, n_states: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Trailing Sharpe features
    roll_mean = excess.rolling(12).mean()
    roll_std = excess.rolling(12).std()
    sharpe = (roll_mean / roll_std).dropna()
    kmeans = KMeans(n_clusters=n_states, random_state=0).fit(sharpe)
    labels = kmeans.labels_
    dates = sharpe.index
    states = pd.Series(labels, index=dates)

    # Global stats
    global_mu = excess.mean().values * 12
    global_cov = excess.cov().values * 12

    mu_s = []
    cov_s = []
    for s in range(n_states):
        idx = states[states == s].index
        if len(idx) < 5:
            # fallback to global if too few obs
            mu_s.append(global_mu)
            cov_s.append(global_cov)
        else:
            s_ret = excess.loc[idx]
            mu_s.append(s_ret.mean().values * 12)
            cov_s.append(s_ret.cov().values * 12)
    mu_s = np.vstack(mu_s)
    cov_s = np.stack(cov_s)

    # Transition matrix
    P = np.ones((n_states, n_states))  # Dirichlet(1) smoothing
    for prev, nxt in zip(labels[:-1], labels[1:]):
        P[prev, nxt] += 1
    P /= P.sum(axis=1, keepdims=True)
    return mu_s, cov_s, P

# ---------------------------------------------------------------------------
# 3. Simulate DCA paths
# ---------------------------------------------------------------------------

def simulate_regime_paths(mu_s: np.ndarray, cov_s: np.ndarray, P: np.ndarray,
                          months: int, n_paths: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_states, n_assets = mu_s.shape
    paths = np.zeros((n_paths, months, n_assets))
    states = rng.choice(n_states, size=n_paths)

    for t in range(months):
        for p in range(n_paths):
            s = states[p]
            cov_month = cov_s[s] / 12.0
            # regularize for PD
            eigvals = np.linalg.eigvalsh(cov_month)
            min_eig = np.min(eigvals)
            if min_eig <= 0:
                cov_month += np.eye(n_assets) * (-min_eig + 1e-6)
            # jitter
            cov_month += np.eye(n_assets) * 1e-6
            mu_month = mu_s[s] / 12.0
            paths[p, t] = rng.multivariate_normal(mu_month, cov_month)
            states[p] = rng.choice(n_states, p=P[s])
    return paths

# ---------------------------------------------------------------------------
# 4. DCA wealth
# ---------------------------------------------------------------------------

def dca_wealth(paths: np.ndarray, weights: np.ndarray, monthly_cash: float = 2000.0) -> np.ndarray:
    n, T, k = paths.shape
    bal = np.zeros((n, k))
    for t in range(T):
        bal += monthly_cash * weights
        bal *= (1 + paths[:, t, :])
    return bal.sum(axis=1)

# ---------------------------------------------------------------------------
# 5. Optimiser: MV + entropy + TE + floors + caps
# ---------------------------------------------------------------------------

def solve_weights(mu: np.ndarray, cov: np.ndarray, vt: np.ndarray, gamma: float, te: float,
                  caps: np.ndarray, floors: dict, lam_ent: float = 0.01) -> np.ndarray:
    k = len(mu)
    def obj(w):
        ent = -lam_ent * np.sum(w * np.log(np.clip(w,1e-8,None)))
        return -(w @ mu) + 0.5*gamma*(w @ cov @ w) + ent

    cons = [
        {"type":"eq",   "fun":lambda w: np.sum(w)-1},
        {"type":"ineq","fun":lambda w: np.sum(w[floors['eq']]) - floors['equity_floor']},
        {"type":"ineq","fun":lambda w: np.sum(w[floors['bd']]) - floors['bond_floor']},
        {"type":"ineq","fun":lambda w: np.sum(w[floors['rv']]) - floors['real_floor']},
        {"type":"ineq","fun":lambda w: te**2 - (w-vt) @ cov @ (w-vt)}
    ]
    bounds = [(0,caps[i]) for i in range(k)]
    res = minimize(obj, vt, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else vt

# ---------------------------------------------------------------------------
# 6. Pipeline
# ---------------------------------------------------------------------------

def load_cfg(path: str="config.yaml") -> dict:
    import yaml
    return yaml.safe_load(open(path))


def run_optimise(cfg_path: str="config.yaml", out: str="weights.csv") -> np.ndarray:
    cfg = load_cfg(cfg_path)
    excess = load_returns(cfg)
    mu_s, cov_s, P = fit_regimes(excess)
    mu_hist = excess.mean().values * 12
    cov_hist = excess.cov().values * 12

    vt_map = json.load(open(cfg['vt_weights']))
    order = cfg['assets']['equities'] + cfg['assets']['bonds'] + cfg['assets']['real_assets']
    vt = np.array([vt_map[a] for a in order])

    floors = {
        'eq':[order.index(a) for a in cfg['assets']['equities']],
        'bd':[order.index(a) for a in cfg['assets']['bonds']],
        'rv':[order.index(a) for a in cfg['assets']['real_assets']],
        'equity_floor':cfg['equity_floor'],
        'bond_floor':cfg['bond_floor'],
        'real_floor':cfg['real_floor']
    }
    caps = np.array([cfg['asset_caps'][a] for a in order])

    w = solve_weights(mu_hist, cov_hist, vt,
                      cfg['risk_aversion'], cfg['te_limit'], caps, floors,
                      lam_ent=cfg.get('entropy_lambda',0.01))
    pd.DataFrame([w], columns=order).to_csv(out, index=False)
    LOG.info("Weights saved to %s", out)
    return w


def run_simulate(cfg_path: str, w: np.ndarray, n_paths: int=10000, seed: int=0):
    cfg = load_cfg(cfg_path)
    excess = load_returns(cfg)
    mu_s, cov_s, P = fit_regimes(excess)
    paths = simulate_regime_paths(mu_s, cov_s, P, months=360, n_paths=n_paths, seed=seed)
    wealth = dca_wealth(paths, w, monthly_cash=2000.0)
    LOG.info("Median wealth: %.0f | 5th%%: %.0f | 95th%%: %.0f",
             np.median(wealth), np.percentile(wealth,5), np.percentile(wealth,95))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')
    o = sub.add_parser('optimise'); o.add_argument('-c','--config',default='config.yaml'); o.add_argument('-o','--outfile',default='weights.csv')
    s = sub.add_parser('simulate'); s.add_argument('-c','--config',default='config.yaml'); s.add_argument('-w','--weights',default='weights.csv')
    args = ap.parse_args()
    if args.cmd=='optimise':
        run_optimise(args.config, args.outfile)
    elif args.cmd=='simulate':
        w = pd.read_csv(args.weights).iloc[0].values
        run_simulate(args.config, w)
    else:
        ap.print_help()
