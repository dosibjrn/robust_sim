import numpy as np

def bl_posterior(mu_eq, mu_hat, cov, sigma, n_obs, tau):
    omega = np.diag((sigma / np.sqrt(n_obs))**2)
    inv   = np.linalg.inv(tau*cov + omega)
    return mu_eq + tau*cov @ inv @ (mu_hat - mu_eq)
