import os
import logging
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
app = FastAPI(title="NiftyEdge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/model_v1.pkl")

THRESHOLD = 0.55
TICKERS   = list(SECTOR_MAP.keys())
NSEI      = "^NSEI"

load_dotenv()
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
openrouter_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# ── Request schema ────────────────────────────────────────────────────────────
class ChatPayload(BaseModel):
    message: str
    signals: list = []
    history: list = []   # [{role: "user"|"assistant", content: "..."}]

# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_recent_data(lookback_days: int = 120):
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)

    raw = yf.download(TICKERS, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), group_by="ticker")
    raw = raw.swaplevel(axis=1).stack(level=1).reset_index()
    raw.columns.name = None
    raw.rename(columns={"level_1": "Ticker"}, inplace=True)
    raw["Date"] = pd.to_datetime(raw["Date"])

    nsei_raw = yf.download(NSEI, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
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


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/signals")
def get_signals():
    try:
        raw, nsei_raw = fetch_recent_data()

        if raw.empty:
            raise HTTPException(status_code=503, detail="Failed to fetch market data")

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

    except Exception as e:
        log.exception("Error in /signals")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(payload: ChatPayload):
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