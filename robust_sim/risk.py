import numpy as np
from sklearn.covariance import LedoitWolf

def calc_cov(excess):
    lw = LedoitWolf().fit(excess)
    cov   = lw.covariance_ * 12
    sigma = np.sqrt(np.diag(cov))
    return cov, sigma, excess.shape[0]

def compute_cvar(excess, weights, alpha):
    port = excess.dot(weights)
    var  = np.quantile(port, 1 - alpha)
    return port[port <= var].mean()
