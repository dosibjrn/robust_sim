#!/usr/bin/env python3
"""
sim_optim.py

Simulation-Based Optimization (SBO) for 30-year DCA portfolio
----------------------------------------------------------------
This script directly optimizes the static weight vector `w` by maximizing the
expected terminal utility under Monte Carlo DCA paths, using a derivative-free
optimizer (Differential Evolution).

Dependencies:
  pip install numpy pandas scipy scikit-learn cvxpy tqdm
"""
import yaml
import numpy as np
import pandas as pd
import logging
from scipy.optimize import differential_evolution
from tqdm import tqdm
# Import existing utility functions from v30 engine
from v30_portfolio_engine import (
    load_returns,
    fit_regimes,
    simulate_regime_paths,
)

# ----------------------------------------
def load_config(path="config.yaml"):
    cfg = yaml.safe_load(open(path))
    logging.getLogger().setLevel(cfg.get("log_level","INFO"))
    return cfg

# ----------------------------------------
# Precompute DCA growth "multipliers" per path/asset

def compute_weights_matrix(paths, monthly_cash=2000):
    # paths: [n_paths, months, n_assets] excess returns
    n_paths, M, K = paths.shape
    # Convert to gross returns
    gross = 1 + paths
    weights_mat = np.zeros((n_paths, K))
    for p in range(n_paths):
        # cumulative product by month
        cumprod = np.cumprod(gross[p], axis=0)  # [M,K]
        final = cumprod[-1]                     # shape [K]
        # previous cumulative (shifted)
        prev = np.vstack([np.ones((1,K)), cumprod[:-1]])  # [M,K]
        # growth factor from t -> end = final / prev[t]
        growth = final / prev                  # [M,K]
        # sum over t of growth factors
        weights_mat[p] = growth.sum(axis=0)
    # scale by monthly_cash
    return monthly_cash * weights_mat  # [n_paths,K]

# ----------------------------------------
# Objective: maximize expected CRRA utility of terminal wealth

def objective(w, weights_mat, gamma):
    # enforce positivity & sum-to-one via normalization
    w = np.maximum(w, 0)
    if w.sum() <= 0:
        return 1e6
    w = w / w.sum()
    # terminal wealth per path = sum_k weights_mat[p,k] * w[k]
    wealth = weights_mat.dot(w)
    # CRRA utility U = W^(1-gamma)/(1-gamma)
    if abs(gamma-1) > 1e-6:
        util = np.mean(wealth**(1-gamma)) / (1-gamma)
    else:
        util = np.mean(np.log(wealth))
    return -util  # negative for minimization

# ----------------------------------------
def run_sbo(config_path: str, output_csv: str):
    cfg = load_config(config_path)
    # 1. load excess returns
    excess = load_returns(cfg)
    # 2. fit regimes
    mu_s, cov_s, P = fit_regimes(excess)
    # 3. simulate paths
    #n_paths = cfg.get("sbo_n_paths", 2000)
    #months  = cfg.get("sbo_months", 360)

    n_paths = cfg.get("sbo_n_paths", 100)
    months  = cfg.get("sbo_months", 360)

    logging.info(f"Simulating {n_paths} paths for {months} months...")
    paths = simulate_regime_paths(mu_s, cov_s, P, months, n_paths, seed=0)
    # 4. precompute weights matrix
    logging.info("Precomputing DCA growth matrix...")
    weights_mat = compute_weights_matrix(paths, monthly_cash=cfg.get("monthly_cash",2000))
    K = paths.shape[2]
    # 5. optimize via Differential Evolution
    gamma = cfg["risk_aversion"]
    bounds = [(0,1)] * K
    logging.info("Starting differential evolution optimization...")
    result = differential_evolution(
        lambda w: objective(w, weights_mat, gamma),
        bounds,
        strategy="best1bin",
        maxiter=100,
        popsize=15,
        tol=1e-3,
        polish=True
    )
    w_opt = np.maximum(result.x, 0)
    w_opt /= w_opt.sum()
    # 6. save
    symbols = list(cfg["assets"]["equities"]) + list(cfg["assets"]["bonds"]) + list(cfg["assets"]["real_assets"])
    df = pd.DataFrame([w_opt], columns=symbols)
    df.to_csv(output_csv, index=False)
    logging.info(f"Simulation-based optimized weights saved to {output_csv}")

# ----------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulation-Based Optimization CLI")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-o", "--output", default="weights_sbo.csv")
    args = parser.parse_args()
    run_sbo(args.config, args.output)
