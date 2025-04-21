import json, numpy as np, pandas as pd, yaml, tempfile
from robust_sim.pipeline import run

def test_vt_invariance():
    vt = json.load(open("data/vt_weights.json"))
    cfg = {
        "assets":{
            "equities":{"US":"^GSPC"},
            "bonds":   {},
        },
        "risk_free_fred":"DGS1",
        "start_date":"2010-01-01",
        "bayesian":False,
        "te_limit":None,
        "equity_floor":0.8,
        "log_level":"CRITICAL"
    }
    with tempfile.NamedTemporaryFile("w",suffix=".yml",delete=False) as f:
        yaml.safe_dump(cfg,f)
        cfg_path=f.name
    weights = run(cfg_path,"tmp.csv",refresh_data=False)
    w = weights[:len(vt)]
    vt_vec = np.array(list(vt.values()))[:len(w)]
    assert np.allclose(w, vt_vec, atol=1e-6)
