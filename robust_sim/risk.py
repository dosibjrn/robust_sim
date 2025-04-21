import numpy as np
from sklearn.covariance import LedoitWolf

def calc_cov(excess):
    """
    Estimate annualized cov via Ledoit–Wolf.
    Returns cov, sigma vector, n_obs.
    """
    lw = LedoitWolf().fit(excess)
    cov = lw.covariance_ * 12
    sigma = np.sqrt(np.diag(cov))
    n_obs = excess.shape[0]
    return cov, sigma, n_obs

def compute_cvar(excess_returns, weights, alpha):
    """
    Empirical CVaR at level alpha from historical excess returns.
    """
    port = excess_returns.dot(weights)
    var = np.quantile(port, 1 - alpha)
    cvar = port[port <= var].mean()
    return cvar
