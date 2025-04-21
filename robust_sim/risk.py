from sklearn.covariance import LedoitWolf
import numpy as np

def calc_cov(excess):
    lw = LedoitWolf().fit(excess)
    cov = lw.covariance_ * 12     # annualise
    sigma = np.sqrt(np.diag(cov))
    return cov, sigma, len(excess)
