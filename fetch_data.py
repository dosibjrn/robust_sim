#!/usr/bin/env python3
"""
fetch_data.py

One‑off script to pull:
  - Equity price history (US, Dev_exUS, EM, BTC)
  - Bond price history (EUNA.DE)
  - Risk‑free 1‑yr Treasury yield (FRED DGS1)
And write static market‑cap weights.

Outputs into ./data/:
  equity_prices.csv
  bond_prices.csv
  rf_real.csv
  vt_weights.json
"""

import os
import json
from datetime import datetime

import pandas as pd
import yfinance as yf
import pandas_datareader.data as web

# Configuration
START_DATE = "1990-01-01"
FRED_RF    = "DGS1"

EQUITY_TICKERS = {
    "US":       "^GSPC",
    "Dev_exUS": "IEV",
    "EM":       "EEM",
    "BTC":      "BTC-USD"
}

BOND_TICKERS = {
    "EUNA": "EUNA.DE"
}

VT_WEIGHTS = {
    "US":       0.60,
    "Dev_exUS": 0.29,
    "EM":       0.11,
    "BTC":      0.00,
    "FinWood":  0.00,
    "EUNA":     0.00
}


def fetch_equities(tickers, start, end):
    df = pd.DataFrame()
    for name, symbol in tickers.items():
        data = yf.download(symbol, start=start, end=end, progress=False)
        df[name] = data["Close"]
    return df


def fetch_bonds(tickers, start, end):
    df = pd.DataFrame()
    for name, symbol in tickers.items():
        data = yf.download(symbol, start=start, end=end, progress=False)
        df[name] = data["Close"]
    return df


def fetch_rf(fred_code, start, end):
    series = web.DataReader(fred_code, "fred", start, end)[fred_code]
    return series / 100.0


def main():
    os.makedirs("data", exist_ok=True)
    start = START_DATE
    end = datetime.today().strftime("%Y-%m-%d")

    print("Fetching equity prices...")
    eq = fetch_equities(EQUITY_TICKERS, start, end)
    eq.to_csv("data/equity_prices.csv", index_label="Date")
    print(f"Wrote {len(eq)} rows to data/equity_prices.csv")

    print("Fetching bond prices...")
    bd = fetch_bonds(BOND_TICKERS, start, end)
    bd.to_csv("data/bond_prices.csv", index_label="Date")
    print(f"Wrote {len(bd)} rows to data/bond_prices.csv")

    print("Fetching risk-free series...")
    rf = fetch_rf(FRED_RF, start, end)
    rf.to_frame("RF").to_csv("data/rf_real.csv", index_label="Date")
    print(f"Wrote {len(rf)} rows to data/rf_real.csv")

    print("Writing vt_weights.json...")
    with open("data/vt_weights.json", "w") as f:
        json.dump(VT_WEIGHTS, f, indent=2)
    print("Wrote data/vt_weights.json")

    print("Data fetch complete. You can now run:")
    print("  python -m robust_sim.cli -c config.yaml -o weights.csv")


if __name__ == "__main__":
    main()
