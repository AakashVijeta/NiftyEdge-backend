# Graph Report - .  (2026-04-25)

## Corpus Check
- Corpus is ~3,212 words - fits in a single context window. You may not need a graph.

## Summary
- 76 nodes · 80 edges · 11 communities detected
- Extraction: 76% EXTRACTED · 24% INFERRED · 0% AMBIGUOUS · INFERRED: 19 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_API Routes & Serving|API Routes & Serving]]
- [[_COMMUNITY_LLM Chat Pipeline|LLM Chat Pipeline]]
- [[_COMMUNITY_Model Training & Backtest|Model Training & Backtest]]
- [[_COMMUNITY_Feature Computations|Feature Computations]]
- [[_COMMUNITY_Data Ingestion & Features|Data Ingestion & Features]]
- [[_COMMUNITY_LLM Provider Config|LLM Provider Config]]
- [[_COMMUNITY_Backtest Helpers|Backtest Helpers]]
- [[_COMMUNITY_Product Overview|Product Overview]]
- [[_COMMUNITY_Ingest Fetch|Ingest Fetch]]
- [[_COMMUNITY_Prediction Script|Prediction Script]]
- [[_COMMUNITY_Training Script|Training Script]]

## God Nodes (most connected - your core abstractions)
1. `main.py (FastAPI)` - 9 edges
2. `features.py` - 6 edges
3. `train.py` - 5 edges
4. `XGBoost Classifier` - 5 edges
5. `build_features()` - 4 edges
6. `models/model_v1.pkl` - 4 edges
7. `POST /chat` - 4 edges
8. `Groq (Llama 3.3 70B)` - 4 edges
9. `OpenRouter Fallback` - 4 edges
10. `.env Environment Config` - 4 edges

## Surprising Connections (you probably didn't know these)
- `features.py` --references--> `pandas 3.0.1`  [INFERRED]
  README.md → requirements.txt
- `features.py` --references--> `numpy 2.4.3`  [INFERRED]
  README.md → requirements.txt
- `train.py` --references--> `scikit-learn 1.8.0`  [INFERRED]
  README.md → requirements.txt
- `main.py (FastAPI)` --references--> `fastapi 0.135.1`  [INFERRED]
  README.md → requirements.txt
- `main.py (FastAPI)` --references--> `pydantic 2.12.5`  [INFERRED]
  README.md → requirements.txt

## Hyperedges (group relationships)
- **Data-to-Serving Pipeline** —  [EXTRACTED 1.00]
- **Chat LLM Stack** —  [EXTRACTED 1.00]
- **Backtest Trade Simulation Config** —  [EXTRACTED 1.00]

## Communities

### Community 0 - "API Routes & Serving"
Cohesion: 0.17
Nodes (13): CORS Allowed Origins, GET /health, GET /signals, NiftyEdge Frontend (Netlify), main.py (FastAPI), models/model_v1.pkl, predict.py, 0.55 Probability Threshold (+5 more)

### Community 1 - "LLM Chat Pipeline"
Cohesion: 0.23
Nodes (10): BaseModel, build_signals_text(), build_system_prompt(), call_llm(), chat(), ChatPayload, fetch_recent_data(), get_signals() (+2 more)

### Community 2 - "Model Training & Backtest"
Cohesion: 0.22
Nodes (11): Backtest Parameters (TP 2.5% / SL 1.0% / 5d), backtest.py, Backtesting Harness, scale_pos_weight Class Imbalance Handling, data/processed/, 5-Day 1.5% Gain Target, train.py, Walk-Forward Validation (+3 more)

### Community 3 - "Feature Computations"
Cohesion: 0.24
Nodes (4): build_features(), compute_rsi(), compute_sector_momentum(), df      : stock OHLCV dataframe with columns [Date, Ticker, Open, High, Low, Clo

### Community 4 - "Data Ingestion & Features"
Cohesion: 0.2
Nodes (10): data/raw/, 12 Engineered Features, features.py, ingest.py, Nifty 50 Universe, Technical Features (MA/RSI/MACD/BB/Volume/RS/Sector), yfinance Data Source, numpy 2.4.3 (+2 more)

### Community 5 - "LLM Provider Config"
Cohesion: 0.32
Nodes (8): POST /chat, .env Environment Config, Groq (Llama 3.3 70B), OpenRouter Fallback, python-dotenv 1.2.2, groq 1.1.1, httpx 0.28.1, openai 2.29.0

### Community 6 - "Backtest Helpers"
Cohesion: 0.5
Nodes (2): get_future_closes(), Get close prices for n trading days after entry_date.

### Community 7 - "Product Overview"
Cohesion: 0.5
Nodes (4): AI Analyst Chat, Daily Signal Generation, Disclaimer (Not Financial Advice), NiftyEdge API

### Community 8 - "Ingest Fetch"
Cohesion: 1.0
Nodes (0): 

### Community 9 - "Prediction Script"
Cohesion: 1.0
Nodes (0): 

### Community 10 - "Training Script"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **26 isolated node(s):** `Get close prices for n trading days after entry_date.`, `df      : stock OHLCV dataframe with columns [Date, Ticker, Open, High, Low, Clo`, `Produce a rich signal table for the system prompt.`, `Try Groq first, fall back to OpenRouter.`, `NiftyEdge Frontend (Netlify)` (+21 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Ingest Fetch`** (2 nodes): `fetch_data()`, `ingest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Prediction Script`** (1 nodes): `predict.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Training Script`** (1 nodes): `train.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `train.py` connect `Model Training & Backtest` to `API Routes & Serving`?**
  _High betweenness centrality (0.190) - this node is a cross-community bridge._
- **Why does `main.py (FastAPI)` connect `API Routes & Serving` to `LLM Provider Config`?**
  _High betweenness centrality (0.176) - this node is a cross-community bridge._
- **Why does `models/model_v1.pkl` connect `API Routes & Serving` to `Model Training & Backtest`?**
  _High betweenness centrality (0.161) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `main.py (FastAPI)` (e.g. with `fastapi 0.135.1` and `pydantic 2.12.5`) actually correct?**
  _`main.py (FastAPI)` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `features.py` (e.g. with `pandas 3.0.1` and `numpy 2.4.3`) actually correct?**
  _`features.py` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `XGBoost Classifier` (e.g. with `Walk-Forward Validation` and `xgboost 3.2.0`) actually correct?**
  _`XGBoost Classifier` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Get close prices for n trading days after entry_date.`, `df      : stock OHLCV dataframe with columns [Date, Ticker, Open, High, Low, Clo`, `Produce a rich signal table for the system prompt.` to the rest of the system?**
  _26 weakly-connected nodes found - possible documentation gaps or missing edges._