import yaml
import jsonschema as js

_SCHEMA = {
    "type":"object",
    "properties":{
        "use_local_data": {"type":"boolean"},
        "assets": {
            "type":"object",
            "properties":{
                "equities":{"type":"array"},
                "bonds":{"type":"array"},
                "real_assets":{"type":"array"}
            },
            "required":["equities","bonds","real_assets"]
        },
        "risk_free_csv":{"type":"string"},
        "equity_csv":   {"type":"string"},
        "bond_csv":     {"type":"string"},
        "wood_csv":     {"type":"string"},
        "vt_weights":   {"type":"string"},
        "risk_aversion":{"type":"number","default":5.0},
        "te_limit":     {"type":["number","null"],"default":0.04},
        "equity_floor":{"type":"number","default":0.8},
        "bond_floor":  {"type":"number","default":0.1},
        "real_floor":  {"type":"number","default":0.0},
        "cvar_threshold":{"type":"number","default":0.95},
        "asset_caps":  {"type":"object"},
        "bayesian":    {"type":"boolean","default":False},
        "shr_tau":     {"type":"number","default":0.025},
        "log_level":   {"type":"string","default":"INFO"}
    },
    "required":[
        "use_local_data","assets","risk_free_csv","equity_csv",
        "bond_csv","wood_csv","vt_weights"
    ]
}

def load(path="config.yaml"):
    cfg = yaml.safe_load(open(path))
    js.validate(cfg, _SCHEMA)
    return cfg
