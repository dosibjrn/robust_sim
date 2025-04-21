import yaml, logging

def load(path="config.yaml"):
    cfg = yaml.safe_load(open(path))
    return cfg
