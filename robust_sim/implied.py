import json
import numpy as np

def implied_mu(cov, delta=3.0, vt_path="data/vt_weights.json"):
    """
    Market-implied equilibrium returns: mu* = delta * cov * vt
    """
    vt = json.load(open(vt_path))
    w  = np.array(list(vt.values()))
    return delta * cov.dot(w)
