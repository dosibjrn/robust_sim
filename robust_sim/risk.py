# robust_sim/risk.py

import numpy as np
from sklearn.covariance import LedoitWolf

def calc_cov(excess):
    """
    Estimate the annualized covariance matrix via Ledoit–Wolf shrinkage,
    and return (covariance, sigma_vector, number_of_observations).

    Parameters
    ----------
    excess : pandas.DataFrame
        Monthly excess returns (assets as columns).

    Returns
    -------
    cov : numpy.ndarray
        Annualized covariance matrix (n_assets x n_assets).
    sigma : numpy.ndarray
        Annualized volatilities (length = n_assets).
    n_obs : int
        Number of monthly observations.
    """
    # Fit Ledoit–Wolf on monthly excess returns
    lw = LedoitWolf().fit(excess)
    cov_month = lw.covariance_
    # Annualize covariance and compute sigma
    cov = cov_month * 12
    sigma = np.sqrt(np.diag(cov))
    n_obs = excess.shape[0]
    return cov, sigma, n_obs
