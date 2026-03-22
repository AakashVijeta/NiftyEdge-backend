from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import joblib
from datetime import datetime, timedelta

from features import build_features, FEATURES, SECTOR_MAP

# ── Startup ───────────────────────────────────────────────────────────────────
app = FastAPI(title="NiftyEdge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/model_v1.pkl")

THRESHOLD = 0.55

TICKERS = list(SECTOR_MAP.keys())
NSEI    = "^NSEI"

# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_recent_data(lookback_days: int = 120):
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)
    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    # ── Stock data ────────────────────────────────────────────────────────────
    raw = yf.download(TICKERS, start=start_str, end=end_str, group_by="ticker")

    # Columns are (Ticker, OHLCV) — swap levels so OHLCV is on top, then stack
    raw = raw.swaplevel(axis=1)         # now (OHLCV, Ticker)
    raw = raw.stack(level=1)            # Ticker becomes a row index level
    raw = raw.reset_index()
    raw.columns.name = None
    raw.rename(columns={"level_1": "Ticker"}, inplace=True)
    raw["Date"] = pd.to_datetime(raw["Date"])

    # ── NSEI index ────────────────────────────────────────────────────────────
    nsei_raw = yf.download(NSEI, start=start_str, end=end_str)
    # Columns are (OHLCV, '^NSEI') — flatten to just OHLCV names
    nsei_raw.columns = [col[0] for col in nsei_raw.columns]
    nsei_raw = nsei_raw.reset_index()[["Date", "Close"]]
    nsei_raw["Date"] = pd.to_datetime(nsei_raw["Date"])

    return raw, nsei_raw

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/signals")
def get_signals():
    try:
        # 1. Fetch recent data
        raw, nsei_raw = fetch_recent_data()

        if raw.empty:
            raise HTTPException(status_code=503, detail="Failed to fetch market data")

        # 2. Sort and compute features
        raw = raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        df  = build_features(raw, nsei_raw)

        # 3. Take only the latest row per ticker
        latest = df.sort_values("Date").groupby("Ticker").tail(1).copy()
        latest = latest.dropna(subset=FEATURES)

        if latest.empty:
            return []

        # 4. Predict
        X      = latest[FEATURES].values
        probas = model.predict_proba(X)[:, 1]
        latest["probability"] = probas

        # 5. Filter by threshold
        signals = latest[latest["probability"] >= THRESHOLD].copy()
        signals["sector"] = signals["Ticker"].map(SECTOR_MAP)
        signals = signals.sort_values("probability", ascending=False)

        # 6. Return clean JSON
        return [
            {
                "ticker":      row["Ticker"],
                "date":        row["Date"].strftime("%Y-%m-%d"),
                "probability": round(float(row["probability"]), 3),
                "sector":      row["sector"]
            }
            for _, row in signals.iterrows()
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))