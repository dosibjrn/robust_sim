import json, numpy as np
from robust_sim.pipeline import run

def test_vt(tmp_path):
    cfg = {
      "use_local_data": True,
      "assets":{"equities":["US"],"bonds":["EUNA"],"wood":["FinWood"]},
      "equity_csv":"data/equity_prices.csv",
      "bond_csv":"data/bond_prices.csv",
      "wood_csv":"data/wood_prices.csv",
      "risk_free_csv":"data/rf_real.csv",
      "vt_weights":"data/vt_weights.json",
      "bayesian": False,
      "te_limit": None,
      "equity_floor": 0.8,
      "risk_aversion": 5,
      "shr_tau": 0.025,
      "log_level":"CRITICAL"
    }
    import yaml; yaml.safe_dump(cfg, open(tmp_path/"cfg.yml","w"))
    weights = run(str(tmp_path/"cfg.yml"), out_csv=str(tmp_path/"w.csv"))
    vt = np.array(list(json.load(open("data/vt_weights.json")).values()))
    assert np.allclose(weights, vt, atol=1e-6)
