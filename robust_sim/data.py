# Updated robust_sim/data.py with proper wood_prices.csv handling

import pandas as pd
from pandas.tseries.offsets import MonthEnd

def prepare_data(cfg, refresh=False):
    """
    Load and preprocess local CSVs:
      - data/equity_prices.csv  (Date index)
      - data/bond_prices.csv    (Date index)
      - data/wood_prices.csv    (Year, Mänty, Kuusi, Koivu)
      - data/rf_real.csv        (Date index)
    Returns:
      excess returns DataFrame with columns US, Dev_exUS, EM, BTC, EUNA, FinWood
    """
    # 1. Read price series
    eq = pd.read_csv("data/equity_prices.csv", parse_dates=["Date"], index_col="Date")
    bd = pd.read_csv("data/bond_prices.csv",    parse_dates=["Date"], index_col="Date")

    # 2. Read and convert annual wood prices to monthly FinWood index
    dfw = pd.read_csv("data/wood_prices.csv")
    dfw["Date"] = pd.to_datetime(dfw["Year"].astype(str), format="%Y") + MonthEnd(0)
    dfw = dfw.set_index("Date")[["Mänty (€)","Kuusi (€)","Koivu (€)"]]
    # Ensure values in euros; assume wood_prices.csv already in €
    # Resample to month-end and interpolate linearly
    dfw_monthly = dfw.resample("M").interpolate()
    # Combine into single FinWood price (average of species)
    finwood = pd.DataFrame({"FinWood": dfw_monthly.mean(axis=1)})

    # 3. Read risk-free series
    rf = pd.read_csv("data/rf_real.csv", parse_dates=["Date"], index_col="Date")["RF"]

    # 4. Combine all price series
    prices = pd.concat([eq, bd, finwood], axis=1).dropna()

    # 5. Resample to month-end
    prices = prices.resample("M").last()
    rf_mon = rf.resample("M").last()

    # 6. Compute returns and excess returns
    ret = prices.pct_change().dropna()
    excess = ret.sub(rf_mon, axis=0).dropna()

    return excess
