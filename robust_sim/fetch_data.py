import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from datetime import datetime

# ---------- helpers ----------
def yf_prices(tickers: dict, start, end):
    df = pd.DataFrame()
    for name, tkr in tickers.items():
        df[name] = yf.download(tkr, start=start, end=end, progress=False)["Adj Close"]
    return df

def fred_series(code: str, start, end):
    s = web.DataReader(code, "fred", start, end)[code]
    return s / 100.0   # convert % → decimal

# ---------- wood ----------
def fetch_wood_local(path="data/wood_prices.csv"):
    """
    Reads your annual Finnish wood price CSV (Year;Mänty;Kuusi;Koivu in €/m³ or mk/m³).
    Returns a monthly DataFrame with a single FinWood column (average of species).
    """
    df = pd.read_csv(path, sep='[;,]', engine="python")
    df['Date'] = pd.to_datetime(df['Year'], format='%Y') + MonthEnd(0)
    df = df.set_index('Date')[['Mänty','Kuusi','Koivu']]
    # convert mk→€ if needed
    if df.max().max() > 1000:
        df /= 5.95
    df_m = df.resample("M").interpolate()
    df_m['FinWood'] = df_m.mean(axis=1)
    return df_m[['FinWood']]

# ---------- main builder ----------
def build_and_save(cfg):
    start, end = cfg["start_date"], datetime.today().strftime("%Y-%m-%d")
    eq_ticks   = cfg["assets"]["equities"]
    bond_ticks = cfg["assets"]["bonds"]

    eq_p   = yf_prices(eq_ticks,   start, end)
    bond_p = yf_prices(bond_ticks, start, end)
    wood_p = fetch_wood_local()

    raw = pd.concat([eq_p, bond_p, wood_p], axis=1).dropna()
    raw.to_csv("data/monthly_raw.csv")

    fred_code = cfg["risk_free_fred"]
    rf = fred_series(fred_code, start, end)
    rf.to_frame("RF").to_csv("data/rf_raw.csv")
