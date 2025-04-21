import yaml, jsonschema as js

_SCHEMA = {
    "type": "object",
    "properties": {
        "assets":        {"type":"object"},
        "wood_url":      {"type":"string"},
        "risk_free_fred":{"type":"string"},
        "start_date":    {"type":"string"},
        "risk_aversion": {"type":"number","default":5.0},
        "te_limit":      {"type":["number","null"],"default":0.04},
        "equity_floor":  {"type":"number","default":0.8},
        "bayesian":      {"type":"boolean","default":False},
        "shr_tau":       {"type":"number","default":0.025},
        "log_level":     {"type":"string","default":"INFO"}
    },
    "required": ["assets","risk_free_fred","start_date"]
}

def load(path="config.yaml"):
    cfg = yaml.safe_load(open(path))
    js.validate(cfg, _SCHEMA)
    return cfg
