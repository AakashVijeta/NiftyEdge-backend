import yfinance as yf
import pandas as pd
import os

NIFTY50_TICKERS = [
    "^NSEI", "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BEL.NS", "BHARTIARTL.NS", "CIPLA.NS", "COALINDIA.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "ETERNAL.NS", "GRASIM.NS",
    "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDIGO.NS", "INFY.NS",
    "ITC.NS", "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS",
    "LT.NS", "M&M.NS", "MARUTI.NS", "MAXHEALTH.NS",
    "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
    "RELIANCE.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATACONSUM.NS", "TMCV.NS",
    "TATASTEEL.NS", "TECHM.NS", "TITAN.NS", "TRENT.NS",
    "ULTRACEMCO.NS", "WIPRO.NS"
]

def fetch_data(start, end):
    df = yf.download(NIFTY50_TICKERS, start=start, end=end, group_by='ticker')
    df = df.stack(level=0).reset_index()
    df.columns.name = None
    df.rename(columns={"level_1": "Ticker"}, inplace=True)
    return df

if __name__ == "__main__":
    start_date = "2018-01-01"
    end_date = "2025-12-31"
    df = fetch_data(start_date, end_date)
    os.makedirs("data/raw", exist_ok=True)
    
    # Separating index from stocks
    nsei = df[df["Ticker"] == "^NSEI"].copy()
    stocks = df[df["Ticker"] != "^NSEI"].copy()

    stocks.to_csv("data/raw/nifty50_ohlcv.csv", index=False)
    nsei.to_csv("data/raw/nifty_index.csv", index=False)

    print(f"Stocks : {len(stocks)} rows → data/raw/nifty50_ohlcv.csv")
    print(f"Index  : {len(nsei)} rows  → data/raw/nifty_index.csv")