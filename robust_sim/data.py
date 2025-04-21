import pandas as pd
from .fetch_data import build_and_save

def prepare_data(cfg, refresh=False):
    if refresh:
        build_and_save(cfg)

    raw = pd.read_csv("data/monthly_raw.csv", parse_dates=["Date"], index_col="Date")
    rf  = pd.read_csv("data/rf_raw.csv",      parse_dates=["Date"], index_col="Date")["RF"]

    prices = raw.resample("M").last()
    rf_mon = rf.resample("M").last()
    ret    = prices.pct_change().dropna()
    excess = ret.sub(rf_mon, axis=0).dropna()

    excess.to_csv("data/monthly_real.csv")
    rf_mon.to_csv("data/rf_real.csv", header=["RF"])
    return excess
