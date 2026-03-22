import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import os
from features import FEATURES

# ── Config ────────────────────────────────────────────────────────────────────
TARGET    = "Target"
THRESHOLD = 0.55
FRICTION  = 0.0015  # 0.15% round trip brokerage + slippage
TAKE_PROFIT = 0.025   # exit when up 2.5%
STOP_LOSS   = 0.010   # exit when down 1.0%

# ── Load features ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/nifty50_features.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
df = df.dropna(subset=FEATURES + [TARGET])

# ── Load raw OHLCV for price simulation ───────────────────────────────────────
raw = pd.read_csv("data/raw/nifty50_ohlcv.csv", parse_dates=["Date"])
raw = raw.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# Build price lookup: (ticker, date) → {Open, Close}
price_lookup = {}
for _, row in raw.iterrows():
    price_lookup[(row["Ticker"], row["Date"])] = {
        "open":  row["Open"],
        "close": row["Close"]
    }

# Build sorted trading calendar for next-day lookup
trading_days = sorted(raw["Date"].unique())
trading_day_idx = {d: i for i, d in enumerate(trading_days)}

def next_trading_day(date):
    idx = trading_day_idx.get(date)
    if idx is None or idx + 1 >= len(trading_days):
        return None
    return trading_days[idx + 1]

def get_future_closes(ticker, entry_date, n=5):
    """Get close prices for n trading days after entry_date."""
    idx = trading_day_idx.get(entry_date)
    if idx is None:
        return []
    closes = []
    for i in range(1, n + 1):
        if idx + i >= len(trading_days):
            break
        future_date = trading_days[idx + i]
        p = price_lookup.get((ticker, future_date))
        if p:
            closes.append(p["close"])
    return closes

# ── Walk-forward backtest ─────────────────────────────────────────────────────
all_trades = []

fold_start = pd.Timestamp("2023-01-01")
fold_size  = pd.DateOffset(months=6)
test_end   = pd.Timestamp("2025-12-31")

fold_num = 1
while fold_start + fold_size <= test_end:
    fold_end = fold_start + fold_size

    train_mask = df["Date"] < fold_start
    test_mask  = (df["Date"] >= fold_start) & (df["Date"] < fold_end)

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        fold_start = fold_end
        continue

    X_tr = df.loc[train_mask, FEATURES].values
    y_tr = df.loc[train_mask, TARGET].values
    X_te = df.loc[test_mask,  FEATURES].values

    scale = (y_tr == 0).sum() / (y_tr == 1).sum()
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_tr, y_tr)

    y_proba = model.predict_proba(X_te)[:, 1]
    test_df = df.loc[test_mask].copy()
    test_df["proba"] = y_proba

    # Simulate trades for every signal
    signals = test_df[test_df["proba"] >= THRESHOLD]
    fold_trades = 0

    for _, row in signals.iterrows():
        ticker    = row["Ticker"]
        sig_date  = row["Date"]
        entry_day = next_trading_day(sig_date)

        if entry_day is None:
            continue

        entry_price = price_lookup.get((ticker, entry_day), {}).get("open")
        if entry_price is None or entry_price == 0:
            continue

        future_closes = get_future_closes(ticker, entry_day)
        if not future_closes:
            continue

        # Walk through each day in order, exit when rule triggers
        exit_return = None
        for close in future_closes:
            daily_return = (close - entry_price) / entry_price
            if daily_return >= TAKE_PROFIT:
                exit_return = TAKE_PROFIT   # took profit
                break
            if daily_return <= -STOP_LOSS:
                exit_return = -STOP_LOSS    # stopped out
                break
        
        # If neither triggered, exit at day 5 close
        if exit_return is None:
            exit_return = (future_closes[-1] - entry_price) / entry_price

        gross_return = exit_return
        net_return   = gross_return - FRICTION

        all_trades.append({
            "fold":      fold_num,
            "ticker":    ticker,
            "date":      sig_date,
            "entry":     entry_price,
            "exit_ret":  exit_return,
            "gross_ret": gross_return,
            "net_ret":   net_return,
            "win":       1 if net_return > 0 else 0
        })
        fold_trades += 1

    print(f"Fold {fold_num} ({fold_start.date()} → {fold_end.date()})  Trades: {fold_trades:,}")
    fold_start = fold_end
    fold_num  += 1

# ── Aggregate results ─────────────────────────────────────────────────────────
trades_df = pd.DataFrame(all_trades)
trades_df = trades_df.dropna(subset=["net_ret", "gross_ret"])

avg_return   = trades_df["net_ret"].mean()
win_rate     = trades_df["win"].mean()
avg_win      = trades_df.loc[trades_df["win"] == 1, "net_ret"].mean()
avg_loss     = trades_df.loc[trades_df["win"] == 0, "net_ret"].mean()
total_trades = len(trades_df)

# Fixed 2% capital allocation per trade
POSITION_SIZE = 0.02
capital = 1.0  # start with 1 unit (e.g. ₹1,00,000)
capital_curve = [capital]

for ret in trades_df["net_ret"]:
    capital = capital + (capital * POSITION_SIZE * ret)
    capital_curve.append(capital)

cumulative   = capital - 1
capital_curve = pd.Series(capital_curve)
rolling_max  = capital_curve.cummax()
drawdown     = (capital_curve - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print(f"\n{'─'*50}")
print(f"Total trades      : {total_trades:,}")
print(f"Win rate          : {win_rate*100:.1f}%")
print(f"Avg return/trade  : {avg_return*100:.2f}%")
print(f"Avg win           : {avg_win*100:.2f}%")
print(f"Avg loss          : {avg_loss*100:.2f}%")
print(f"Cumulative return : {cumulative*100:.1f}%  (2% position sizing)")
print(f"Max drawdown      : {max_drawdown*100:.1f}%")
print(f"{'─'*50}")

# Save trades for inspection
os.makedirs("data", exist_ok=True)
trades_df.to_csv("data/backtest_trades.csv", index=False)
print(f"Trade log saved → data/backtest_trades.csv")