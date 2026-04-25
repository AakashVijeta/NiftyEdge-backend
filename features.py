import pandas as pd
import os

SECTOR_MAP = {
    "HDFCBANK.NS": "Finance", "ICICIBANK.NS": "Finance", "KOTAKBANK.NS": "Finance",
    "AXISBANK.NS": "Finance", "SBIN.NS": "Finance", "BAJFINANCE.NS": "Finance",
    "BAJAJFINSV.NS": "Finance", "HDFCLIFE.NS": "Finance", "SBILIFE.NS": "Finance",
    "SHRIRAMFIN.NS": "Finance", "JIOFIN.NS": "Finance",

    "INFY.NS": "IT", "TCS.NS": "IT", "HCLTECH.NS": "IT",
    "WIPRO.NS": "IT", "TECHM.NS": "IT",

    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy", "NTPC.NS": "Energy",
    "POWERGRID.NS": "Energy", "COALINDIA.NS": "Energy", "BEL.NS": "Energy",

    "SUNPHARMA.NS": "Pharma", "CIPLA.NS": "Pharma", "DRREDDY.NS": "Pharma",
    "APOLLOHOSP.NS": "Pharma", "MAXHEALTH.NS": "Pharma",

    "TATASTEEL.NS": "Metals", "JSWSTEEL.NS": "Metals", "HINDALCO.NS": "Metals",

    "MARUTI.NS": "Auto", "BAJAJ-AUTO.NS": "Auto", "EICHERMOT.NS": "Auto",
    "M&M.NS": "Auto", "TMCV.NS": "Auto",

    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "TATACONSUM.NS": "FMCG",

    "ASIANPAINT.NS": "Consumer", "TITAN.NS": "Consumer", "TRENT.NS": "Consumer",
    "ETERNAL.NS": "Consumer",

    "LT.NS": "Infra", "ULTRACEMCO.NS": "Infra", "GRASIM.NS": "Infra",
    "ADANIENT.NS": "Infra", "ADANIPORTS.NS": "Infra",

    "BHARTIARTL.NS": "Telecom",
    "INDIGO.NS": "Aviation",
}

FEATURES = [
    "MA_20", "MA_50", "RSI", "MACD", "Signal",
    "BB_Upper", "BB_Lower", "BB_Position",
    "Volume_Ratio", "Return_5d", "RS_vs_Nifty",
    "Sector_Momentum"
]

def compute_rsi(x):
    delta = x.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def _add_macd_cols(group: pd.DataFrame) -> pd.DataFrame:
    close = group["Close"]
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = ema12 - ema26
    group = group.copy()
    group["MACD"]   = macd
    group["Signal"] = macd.ewm(span=9).mean()
    return group

def _add_bb_cols(group: pd.DataFrame) -> pd.DataFrame:
    close = group["Close"]
    ma20  = close.rolling(20).mean()
    std   = close.rolling(20).std()
    group = group.copy()
    group["BB_Upper"] = ma20 + 2 * std
    group["BB_Lower"] = ma20 - 2 * std
    return group

def calculate_target(group):
    close = group['Close'].values
    targets = []
    for i in range(len(close)):
        if i + 5 >= len(close):
            targets.append(None)
            continue
        future = close[i+1:i+6]
        best = (max(future) - close[i]) / close[i] * 100
        targets.append(1 if best >= 1.5 else 0)
    return pd.Series(targets, index=group.index)

def compute_sector_momentum(df):
    df["_return_10d"] = df.groupby("Ticker")["Close"].transform(
        lambda x: (x / x.shift(10)) - 1
    )
    df["Sector"] = df["Ticker"].map(SECTOR_MAP)
    sector_momentum = (
        df.groupby(["Date", "Sector"])["_return_10d"]
        .mean()
        .reset_index()
        .rename(columns={"_return_10d": "Sector_Momentum"})
    )
    df = df.merge(sector_momentum, on=["Date", "Sector"], how="left")
    df = df.drop(columns=["_return_10d", "Sector"])
    return df

def build_features(df, nsei_df):
    """
    df      : stock OHLCV dataframe with columns [Date, Ticker, Open, High, Low, Close, Volume]
    nsei_df : index OHLCV dataframe with columns [Date, Close]
    Returns df with all feature columns added.
    """
    # Merge NSEI RSI
    nsei_df = nsei_df.copy()
    nsei_df["NSEI_RSI"] = compute_rsi(nsei_df["Close"])
    nsei_df = nsei_df[["Date", "NSEI_RSI"]]
    df = df.merge(nsei_df, on="Date", how="left")

    # Moving averages
    df["MA_20"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
    df["MA_50"] = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(50).mean())

    # RSI
    df["RSI"] = df.groupby("Ticker")["Close"].transform(compute_rsi)

    # MACD and Signal
    df["MACD"] = df.groupby("Ticker")["Close"].transform(
        lambda x: x.ewm(span=12).mean() - x.ewm(span=26).mean()
    )
    df["Signal"] = df.groupby("Ticker")["MACD"].transform(lambda x: x.ewm(span=9).mean())

    # Bollinger Bands
    _ma20 = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
    _std20 = df.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).std())
    df["BB_Upper"] = _ma20 + 2 * _std20
    df["BB_Lower"] = _ma20 - 2 * _std20
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    # Volume ratio
    df["Volume_Ratio"] = df.groupby("Ticker")["Volume"].transform(
        lambda x: x / x.rolling(20).mean()
    )

    # 5-day return
    df["Return_5d"] = df.groupby("Ticker")["Close"].transform(
        lambda x: (x / x.shift(5)) - 1
    )

    # Relative strength vs index
    df["RS_vs_Nifty"] = df["RSI"] - df["NSEI_RSI"]

    # Sector momentum
    df = compute_sector_momentum(df)

    return df

if __name__ == "__main__":
    df = pd.read_csv('./data/raw/nifty50_ohlcv.csv')
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"])

    nsei = pd.read_csv('./data/raw/nifty_index.csv')
    nsei["Date"] = pd.to_datetime(nsei["Date"])
    nsei = nsei.sort_values("Date")

    clean_df = build_features(df, nsei)

    # Target is only needed for training, not prediction
    clean_df["Target"] = clean_df.groupby("Ticker", group_keys=False).apply(calculate_target)

    clean_df.dropna(inplace=True)
    os.makedirs("data/processed", exist_ok=True)
    clean_df.to_csv('./data/processed/nifty50_features.csv', index=False)
    print(f"Saved {len(clean_df)} rows with {clean_df.shape[1]} columns")