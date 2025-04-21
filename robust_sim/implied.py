# robust_sim/implied.py

import json
import numpy as np

def implied_mu(cov, delta=3.0):
    """
    Compute market‑implied equilibrium returns:
      μ* = δ · Σ · w_VT

    Parameters
    ----------
    cov : np.ndarray
        Annualized covariance matrix (n_assets × n_assets).
    delta : float, optional
        Investor risk‑aversion scalar (default=3.0).

    Returns
    -------
    mu_eq : np.ndarray
        Equilibrium return vector (length = n_assets).
    """
    # Load VT weights from data/vt_weights.json
    vt = json.load(open("data/vt_weights.json"))
    w  = np.array(list(vt.values()))
    # Compute δ Σ w
    return delta * cov.dot(w)
