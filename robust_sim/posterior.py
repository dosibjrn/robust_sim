import numpy as np

def calibrate_tau(mu_eq, cov):
    """
    Placeholder: return shrinkage tau that roughly aligns
    posterior Sharpe with equilibrium Sharpe.
    """
    # Equilibrium Sharpe
    sr_eq = mu_eq.dot(mu_eq) / np.sqrt(mu_eq.dot(cov).dot(mu_eq))
    # Map to tau in [0.01,0.1]
    return max(0.01, min(0.1, sr_eq / 10))

def bl_posterior(mu_eq, mu_hat, cov, sigma, n_obs, tau):
    """
    Black–Litterman with P=I, Omega = sigma^2 / n_obs.
    """
    omega = np.diag((sigma / np.sqrt(n_obs))**2)
    inv = np.linalg.inv(tau * cov + omega)
    return mu_eq + tau * cov.dot(inv).dot(mu_hat - mu_eq)
