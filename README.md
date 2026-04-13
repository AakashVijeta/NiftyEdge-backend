# NiftyEdge API

A machine-learning powered trading signal API for Nifty 50 stocks. NiftyEdge trains an XGBoost classifier on technical features to identify short-term (5-day) swing trade opportunities, and exposes the signals along with an LLM-backed analyst chat endpoint through FastAPI.

Frontend: [niftyedge.netlify.app](https://niftyedge.netlify.app)

---

## Features

- **Daily signal generation** — predicts the probability that each Nifty 50 stock will gain ≥1.5% within the next 5 trading days.
- **Rich technical features** — MA(20/50), RSI, MACD, Bollinger Bands, volume ratio, relative strength vs. Nifty, sector momentum.
- **Walk-forward validated model** — XGBoost classifier trained with rolling 6-month out-of-sample folds.
- **AI analyst chat** — Groq (Llama 3.3 70B) with OpenRouter fallback, primed with the current day's signals and a structured system prompt.
- **Full backtesting harness** — simulates next-day open entries with 2.5% take-profit / 1.0% stop-loss, including friction costs.

---

## Architecture

```
yfinance ──► ingest.py ──► data/raw/
                              │
                              ▼
                        features.py ──► data/processed/
                              │
                              ▼
                          train.py ──► models/model_v1.pkl
                              │
                              ▼
                        main.py (FastAPI)
                         /           \
                   GET /signals     POST /chat
                                     │
                                     ▼
                              Groq → OpenRouter
```

---

## API Endpoints

### `GET /health`
Liveness probe.
```json
{ "status": "ok" }
```

### `GET /signals`
Returns today's trade signals with model probability ≥ 0.55.
```json
[
  {
    "ticker": "RELIANCE.NS",
    "date": "2026-04-13",
    "probability": 0.712,
    "sector": "Energy",
    "rsi": 54.2,
    "volume_ratio": 1.63,
    "bb_position": 0.45,
    "sector_momentum": 0.018,
    "rs_vs_nifty": 3.1
  }
]
```

### `POST /chat`
Conversational analyst over today's signals.

Request:
```json
{
  "message": "Which of these has the cleanest setup?",
  "signals": [ /* output from /signals */ ],
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Response:
```json
{ "response": "RELIANCE has the cleanest setup — 71% confidence, RSI 54 (momentum zone), 1.6x volume, BB 0.45..." }
```

---

## Model

- **Algorithm:** XGBoost classifier (`n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`)
- **Target:** 1 if max close within next 5 days is ≥1.5% above today's close, else 0
- **Class imbalance:** handled via `scale_pos_weight`
- **Validation:** walk-forward folds, 6 months each, from 2023-01-01 onward
- **Threshold:** 0.55 (signals emitted only above this probability)

### Features (12)
`MA_20`, `MA_50`, `RSI`, `MACD`, `Signal`, `BB_Upper`, `BB_Lower`, `BB_Position`, `Volume_Ratio`, `Return_5d`, `RS_vs_Nifty`, `Sector_Momentum`

### Universe
50 Nifty 50 tickers across Finance, IT, Energy, Pharma, Metals, Auto, FMCG, Consumer, Infra, Telecom, and Aviation (see [features.py](features.py#L4) for the full sector map).

---

## Project Layout

| File | Purpose |
|---|---|
| [main.py](main.py) | FastAPI app — `/health`, `/signals`, `/chat` |
| [features.py](features.py) | Feature engineering + sector mapping |
| [ingest.py](ingest.py) | Fetch raw OHLCV from Yahoo Finance |
| [train.py](train.py) | Walk-forward training + final model export |
| [backtest.py](backtest.py) | Trade simulation with TP/SL and friction |
| [predict.py](predict.py) | Offline prediction helper |
| `models/model_v1.pkl` | Serialized XGBoost model |
| `data/raw/` | Source OHLCV + index data |
| `data/processed/` | Feature matrix used for training |

---

## Setup

### Prerequisites
- Python 3.11+
- API keys for [Groq](https://console.groq.com/) and [OpenRouter](https://openrouter.ai/) (for `/chat`)

### Install
```bash
git clone https://github.com/<you>/NiftyEdge-api.git
cd NiftyEdge-api
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
pip install -r requirements.txt
```

### Environment
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key
OPENROUTER_API_KEY=your_openrouter_key
```

### Run the API
```bash
uvicorn main:app --reload --port 8000
```
Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Retraining the Model

```bash
python ingest.py         # fetches Nifty 50 + ^NSEI OHLCV → data/raw/
python features.py       # builds feature matrix → data/processed/
python train.py          # walk-forward validation + models/model_v1.pkl
python backtest.py       # simulates trades, writes data/backtest_trades.csv
```

### Backtest Parameters
- Entry: next trading day's open after signal
- Take profit: +2.5%
- Stop loss: −1.0%
- Horizon: 5 trading days (exit at day-5 close if neither TP/SL hit)
- Friction: 0.15% round-trip (brokerage + slippage)
- Position sizing: fixed 2% of capital per trade

---

## CORS

Allowed origins are configured in [main.py:23](main.py#L23):
- `https://niftyedge.netlify.app`
- `http://localhost:5173`

Add your own frontend origin there if needed.

---

## Disclaimer

NiftyEdge is a research and educational tool. Signals are probabilistic model outputs, not financial advice. Past backtest performance does not guarantee future returns. Use at your own risk.
