import pandas as pd
from xgboost import XGBClassifier
import joblib, os
from features import FEATURES

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("./data/processed/nifty50_features.csv", parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)

TARGET = "Target"

df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES].values
y = df[TARGET].values

# ── Walk-forward validation ───────────────────────────────────────────────────

df["Date"] = pd.to_datetime(df["Date"])
fold_results = []

fold_start = pd.Timestamp("2023-01-01")
fold_size = pd.DateOffset(months=6)
test_end = pd.Timestamp("2025-12-31")

fold_num = 1
while fold_start + fold_size <= test_end:
    fold_end = fold_start + fold_size

    train_mask = df["Date"] < fold_start
    test_mask  = (df["Date"] >= fold_start) & (df["Date"] < fold_end)

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        fold_start = fold_end
        continue

    X_tr, y_tr = df.loc[train_mask, FEATURES].values, df.loc[train_mask, TARGET].values
    X_te, y_te = df.loc[test_mask,  FEATURES].values, df.loc[test_mask,  TARGET].values

    scale = (y_tr == 0).sum() / (y_tr == 1).sum()
    m = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42,
    )
    m.fit(X_tr, y_tr)

    y_proba = m.predict_proba(X_te)[:, 1]
    y_pred  = (y_proba >= 0.55).astype(int)

    tp = ((y_pred == 1) & (y_te == 1)).sum()
    fp = ((y_pred == 1) & (y_te == 0)).sum()
    fn = ((y_pred == 0) & (y_te == 1)).sum()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    signals = tp + fp

    print(f"Fold {fold_num} ({fold_start.date()} → {fold_end.date()})  "
          f"Precision: {prec:.3f}  Recall: {rec:.3f}  Signals: {signals:,}")

    fold_results.append({"precision": prec, "recall": rec, "signals": signals})
    fold_start = fold_end
    fold_num += 1

avg_prec = sum(r["precision"] for r in fold_results) / len(fold_results)
avg_rec  = sum(r["recall"]    for r in fold_results) / len(fold_results)
print(f"\nAverage across folds — Precision: {avg_prec:.3f}  Recall: {avg_rec:.3f}")

# ── Final model on all data ───────────────────────────────────────────────────
print("\nTraining final model on all data...")
X_all = df[FEATURES].values
y_all = df[TARGET].values
scale = (y_all == 0).sum() / (y_all == 1).sum()

model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    scale_pos_weight=scale,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_all, y_all)
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model_v1.pkl")
print("Model saved → models/model_v1.pkl")