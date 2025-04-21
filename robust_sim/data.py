import pandas as pd
from pandas.tseries.offsets import MonthEnd

def prepare_data(cfg, refresh=False):
    # Load price CSVs
    eq = pd.read_csv(cfg["equity_csv"], parse_dates=["Date"], index_col="Date")
    bd = pd.read_csv(cfg["bond_csv"],   parse_dates=["Date"], index_col="Date")
    wd = pd.read_csv(cfg["wood_csv"])
    wd["Date"] = pd.to_datetime(wd["Year"].astype(str), format="%Y") + MonthEnd(0)
    wd = wd.set_index("Date")[["Mänty (€)","Kuusi (€)","Koivu (€)"]]
    wd_m = wd.resample("ME").interpolate()
    finwood = pd.DataFrame({"FinWood": wd_m.mean(axis=1)})

    prices = pd.concat([eq, bd, finwood], axis=1).dropna()
    prices = prices.resample("ME").last()

    rf  = pd.read_csv(cfg["risk_free_csv"], parse_dates=["Date"], index_col="Date")["RF"]
    rf_m = rf.resample("ME").last()

    # Compute returns without fill
    ret    = prices.pct_change(fill_method=None).dropna()
    excess = ret.sub(rf_m, axis=0).dropna()

    return excess
