import pandas as pd
from pandas.tseries.offsets import MonthEnd

def prepare_data(cfg, refresh=False):
    """
    Load local CSVs, build monthly excess returns DataFrame.
    """
    # Load raw price series
    eq = pd.read_csv(cfg["equity_csv"], parse_dates=["Date"], index_col="Date")
    bd = pd.read_csv(cfg["bond_csv"],   parse_dates=["Date"], index_col="Date")
    wd_annual = pd.read_csv(cfg["wood_csv"])
    # Convert annual wood->monthly
    wd_annual["Date"] = pd.to_datetime(wd_annual["Year"].astype(str), format="%Y") + MonthEnd(0)
    wd = wd_annual.set_index("Date")[["Mänty (€)","Kuusi (€)","Koivu (€)"]]
    wd_monthly = wd.resample("M").interpolate()
    finwood = pd.DataFrame({"FinWood": wd_monthly.mean(axis=1)})

    # Combine price series
    prices = pd.concat([eq, bd, finwood], axis=1).dropna()

    # Resample to month-end
    prices = prices.resample("M").last()
    rf = pd.read_csv(cfg["risk_free_csv"], parse_dates=["Date"], index_col="Date")["RF"]
    rf_mon = rf.resample("M").last()

    # Compute returns and excess returns
    ret = prices.pct_change().dropna()
    excess = ret.sub(rf_mon, axis=0).dropna()

    return excess
