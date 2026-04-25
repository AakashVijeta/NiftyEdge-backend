import os
import time
import logging
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
from groq import Groq
from openai import OpenAI
import pandas as pd
import joblib
from datetime import datetime, timedelta
from features import build_features, FEATURES, SECTOR_MAP
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Startup ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=_warmup_cache, daemon=True)
    t.start()
    yield

app = FastAPI(title="NiftyEdge API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://niftyedge.netlify.app",
                    "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/model_v1.pkl")

THRESHOLD  = 0.55
TICKERS    = list(SECTOR_MAP.keys())
NSEI       = "^NSEI"

load_dotenv()
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CACHE_TTL          = int(os.getenv("SIGNALS_CACHE_TTL", 900))  # seconds; default 15 min

groq_client = Groq(api_key=GROQ_API_KEY)
openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

_cache_lock       = threading.Lock()
_signals_cache: dict = {"data": None, "ts": 0.0}
_refresh_running  = False

# ── Request schema ────────────────────────────────────────────────────────────
class ChatPayload(BaseModel):
    message: str
    signals: list = []
    history: list = []   # [{role: "user"|"assistant", content: "..."}]

# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_recent_data(lookback_days: int = 120):
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    raw = yf.download(TICKERS, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), group_by="ticker", progress=False, threads=True)
    raw = raw.swaplevel(axis=1).stack(level=1).reset_index()
    raw.columns.name = None
    raw.rename(columns={"level_1": "Ticker"}, inplace=True)
    raw["Date"] = pd.to_datetime(raw["Date"])

    nsei_raw = yf.download(NSEI, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
    nsei_raw.columns = [col[0] for col in nsei_raw.columns]
    nsei_raw = nsei_raw.reset_index()[["Date", "Close"]]
    nsei_raw["Date"] = pd.to_datetime(nsei_raw["Date"])

    return raw, nsei_raw


def build_signals_text(signals: list) -> str:
    """Produce a rich signal table for the system prompt."""
    if not signals:
        return "No signals above threshold today."

    lines = []
    for s in signals:
        prob     = round(s.get("probability", 0) * 100, 1)
        rsi      = s.get("rsi")
        vol      = s.get("volume_ratio")
        bb       = s.get("bb_position")
        sec_mom  = s.get("sector_momentum")
        rs       = s.get("rs_vs_nifty")

        # Qualitative tags
        tags = []
        if rsi is not None:
            if rsi < 40:   tags.append("oversold")
            elif rsi > 70: tags.append("overbought⚠")
            else:          tags.append("momentum-ok")
        if vol is not None and vol >= 1.5:
            tags.append(f"vol-spike({vol}x)")
        if bb is not None and bb > 0.8:
            tags.append("extended-entry⚠")
        if sec_mom is not None and sec_mom > 0:
            tags.append("sector-tailwind")
        elif sec_mom is not None:
            tags.append("sector-headwind")

        lines.append(
            f"  {s['ticker']:12s} | {s.get('sector','?'):18s} | conf {prob:4.1f}% "
            f"| RSI {rsi or '—':5} | vol {vol or '—':5} | BB {bb or '—':5} "
            f"| secMom {sec_mom or '—':7} | vsNifty {rs or '—'} | [{', '.join(tags)}]"
        )

    header = (
        f"  {'TICKER':12s} | {'SECTOR':18s} | {'CONF':8s} "
        f"| {'RSI':5s} | {'VOL':5s} | {'BB':5s} | {'SECMOM':7s} | {'vsNIFTY':8s}"
    )
    return header + "\n" + "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """You are NiftyEdge's AI analyst — an expert in Nifty 50 5-day swing trading.

═══════════════════════════════════════
TODAY'S MODEL SIGNALS  ({n} above 55% threshold)
Date: {date}
═══════════════════════════════════════
{signals_table}

═══════════════════════════════════════
SIGNAL FIELD GUIDE
═══════════════════════════════════════
• Confidence   — model probability that stock gains ≥1.5% within 5 trading days
• RSI          — 45–60 ideal momentum zone | <40 oversold | >70 overbought / risky entry
• Volume ratio — vs 20-day avg. ≥1.5x = institutional confirmation. <0.8x = weak conviction
• BB position  — 0 = lower Bollinger Band, 1 = upper. >0.8 = extended, poor risk/reward entry
• Sector mom   — 10-day avg return of sector peers. Positive = tailwind; negative = headwind
• vs Nifty     — stock RSI minus index RSI. Positive = relative strength vs benchmark

═══════════════════════════════════════
TRADE PARAMETERS
═══════════════════════════════════════
• Target  : +2.5%  (5-day horizon)
• Stop    : −1.0%  (hard stop)
• R:R     : 2.5:1

═══════════════════════════════════════
RANKING LOGIC (use when recommending)
═══════════════════════════════════════
Best setups combine: high confidence + RSI 45–60 + vol ≥1.5x + BB <0.8 + positive sector mom
Avoid or flag: RSI >70, BB >0.8, volume <0.8x, negative sector momentum, vol-spike without RSI support

═══════════════════════════════════════
ANALYST RULES
═══════════════════════════════════════
- Be direct and specific. Reference actual tickers, numbers, and field values in your answers.
- Never give generic disclaimers. This is a trading tool, not a financial advice chatbot.
- If asked to rank, compare or shortlist — do it with clear reasoning tied to the data above.
- If a signal looks risky (⚠ tags), say so and explain why.
- Keep responses to 3–5 sentences unless a detailed breakdown is explicitly requested.
- Speak like a sharp desk analyst, not a customer service bot."""


def build_system_prompt(signals: list) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        n=len(signals),
        date=datetime.today().strftime("%d %b %Y, %A"),
        signals_table=build_signals_text(signals),
    )


def call_llm(system: str, history: list, message: str) -> str:
    """Try Groq first, fall back to OpenRouter."""
    messages = [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": message},
    ]

    # ── Groq ──────────────────────────────────────────────────────────────────
    try:
        log.info("Trying Groq (llama-3.3-70b)…")
        res = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.4,      # lower = more consistent analyst tone
            max_tokens=600,
        )
        log.info("Groq success")
        return res.choices[0].message.content.strip()

    except Exception as e:
        log.warning(f"Groq failed: {e}")

    # ── OpenRouter fallback ────────────────────────────────────────────────────
    try:
        log.info("Falling back to OpenRouter…")
        res = openrouter_client.chat.completions.create(
            model="openrouter/auto",
            messages=messages,
            temperature=0.4,
            max_tokens=600,
        )
        log.info("OpenRouter success")
        return res.choices[0].message.content.strip()

    except Exception as e2:
        log.error(f"OpenRouter also failed: {e2}")
        raise RuntimeError("Both LLM providers unavailable.") from e2


def _compute_signals() -> list | None:
    """Fetch data, build features, run model. Returns signal list or None on data failure."""
    raw, nsei_raw = fetch_recent_data()
    if raw.empty:
        return None

    raw    = raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    df     = build_features(raw, nsei_raw)
    latest = df.sort_values("Date").groupby("Ticker").tail(1).copy()
    latest = latest.dropna(subset=FEATURES)

    if latest.empty:
        return []

    X      = latest[FEATURES].values
    probas = model.predict_proba(X)[:, 1]
    latest["probability"] = probas

    signals = latest[latest["probability"] >= THRESHOLD].copy()
    signals["sector"] = signals["Ticker"].map(SECTOR_MAP)
    signals = signals.sort_values("probability", ascending=False)

    return [
        {
            "ticker":          row["Ticker"],
            "date":            row["Date"].strftime("%Y-%m-%d"),
            "probability":     round(float(row["probability"]), 3),
            "sector":          row["sector"],
            "rsi":             round(float(row["RSI"]), 2)             if pd.notna(row.get("RSI"))             else None,
            "volume_ratio":    round(float(row["Volume_Ratio"]), 3)    if pd.notna(row.get("Volume_Ratio"))    else None,
            "bb_position":     round(float(row["BB_Position"]), 3)     if pd.notna(row.get("BB_Position"))     else None,
            "sector_momentum": round(float(row["Sector_Momentum"]), 4) if pd.notna(row.get("Sector_Momentum")) else None,
            "rs_vs_nifty":     round(float(row["RS_vs_Nifty"]), 2)     if pd.notna(row.get("RS_vs_Nifty"))     else None,
        }
        for _, row in signals.iterrows()
    ]


def _warmup_cache():
    global _refresh_running
    try:
        log.info("Refreshing signals cache…")
        result = _compute_signals()
        if result is not None:
            with _cache_lock:
                _signals_cache["data"] = result
                _signals_cache["ts"]   = time.time()
            log.info(f"Signals cache refreshed: {len(result)} signal(s)")
        else:
            log.warning("Refresh: no market data returned")
    except Exception as e:
        log.warning(f"Cache refresh failed: {e}")
    finally:
        with _cache_lock:
            _refresh_running = False


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


def _trigger_background_refresh():
    global _refresh_running
    with _cache_lock:
        if _refresh_running:
            return
        _refresh_running = True
    threading.Thread(target=_warmup_cache, daemon=True).start()


@app.get("/signals")
def get_signals():
    try:
        now = time.time()
        with _cache_lock:
            cached_data = _signals_cache["data"]
            cache_age   = now - _signals_cache["ts"]

        if cached_data is not None:
            if cache_age < CACHE_TTL:
                log.info("Returning cached signals")
            else:
                log.info("Cache stale — returning stale data, refreshing in background")
                _trigger_background_refresh()
            return cached_data

        # Cache empty: block on first compute
        log.info("Cache empty — computing signals synchronously")
        result = _compute_signals()
        if result is None:
            raise HTTPException(status_code=503, detail="Failed to fetch market data")

        with _cache_lock:
            _signals_cache["data"] = result
            _signals_cache["ts"]   = time.time()

        return result

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Error in /signals")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(payload: ChatPayload):
    try:
        system   = build_system_prompt(payload.signals)
        response = call_llm(system, payload.history, payload.message)
        return {"response": response}

    except RuntimeError as e:
        log.error(f"/chat LLM failure: {e}")
        return {"response": "Both AI providers are currently unavailable. Please try again in a moment."}

    except Exception as e:
        log.exception("/chat unexpected error")
        raise HTTPException(status_code=500, detail=str(e))