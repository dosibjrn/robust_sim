import json, numpy as np
def implied_mu(cov, delta=3):
    vt = json.load(open("data/vt_weights.json"))
    w  = np.array(list(vt.values()))
    return delta * cov @ w
