#!/usr/bin/env python3
"""
NIFTY ORACLE — free-data intraday forecast engine for BankNifty & Nifty50.

What it does
------------
1. Pulls FREE data from Yahoo Finance (2y daily + 60d of 15-minute candles).
2. Trains a walk-forward ML model (gradient boosting) that predicts the
   probability that today's session closes above its open (daily bias).
3. Combines that bias with an Opening Range Breakout (ORB) plan:
   entry, stop-loss, target and position size for your capital.
4. Backtests the exact same rules honestly - no lookahead, with costs -
   and prints the REAL win rate. (Spoiler: it is not 100%. Nothing is.)

Usage
-----
  python3 nifty_oracle.py forecast [--capital 100000] [--risk 1.0]
  python3 nifty_oracle.py backtest

The edge is not the win rate. The edge is: risk 1R to make 1.5R+, size
positions so a loss never hurts, and skip low-confidence days.
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

TICKERS = {"BANKNIFTY": "^NSEBANK", "NIFTY50": "^NSEI"}
CONF_THRESHOLD = 0.58   # skip the trade if model confidence is below this
TARGET_R = 1.5          # target = 1.5x the risk
COST_PCT = 0.0004       # ~0.04% round trip for slippage + charges on futures/options proxy


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def fetch(symbol: str):
    """Return (daily 2y, intraday 15m 60d) OHLC frames with flat columns."""
    daily = yf.download(symbol, period="2y", interval="1d", progress=False, auto_adjust=True)
    intra = yf.download(symbol, period="60d", interval="15m", progress=False, auto_adjust=True)
    for df in (daily, intra):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    return daily.dropna(), intra.dropna()


# ----------------------------------------------------------------------
# Features + model (daily bias)
# ----------------------------------------------------------------------

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Features known at (or just after) the open of day t; label = close>open of day t."""
    f = pd.DataFrame(index=daily.index)
    c, o, h, l = daily["Close"], daily["Open"], daily["High"], daily["Low"]

    f["ret1"] = c.pct_change().shift(1)
    f["ret3"] = c.pct_change(3).shift(1)
    f["ret5"] = c.pct_change(5).shift(1)
    f["rsi"] = rsi(c).shift(1)
    f["ema20_dist"] = (c / c.ewm(span=20).mean() - 1).shift(1)
    f["ema50_dist"] = (c / c.ewm(span=50).mean() - 1).shift(1)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = (tr.ewm(alpha=1 / 14, adjust=False).mean() / c).shift(1)
    f["range_pos"] = ((c - l.rolling(10).min()) /
                      (h.rolling(10).max() - l.rolling(10).min())).shift(1)
    f["gap"] = o / c.shift(1) - 1          # known at 09:15
    f["dow"] = f.index.dayofweek
    f["label"] = (c > o).astype(int)       # intraday direction of day t
    return f.dropna()


def walk_forward(feats: pd.DataFrame, test_days: int = 150):
    """Expanding-window walk-forward predictions - no lookahead."""
    cols = [c for c in feats.columns if c != "label"]
    X, y = feats[cols].values, feats["label"].values
    n = len(feats)
    start = max(n - test_days, 60)
    rows = []
    model = None
    for i in range(start, n):
        if model is None or (i - start) % 10 == 0:   # refit every 10 days
            model = GradientBoostingClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42)
            model.fit(X[:i], y[:i])
        rows.append((feats.index[i], model.predict_proba(X[i:i + 1])[0, 1], y[i]))
    out = pd.DataFrame(rows, columns=["date", "p_up", "label"]).set_index("date")
    return out, model, cols


def fit_full(feats: pd.DataFrame):
    cols = [c for c in feats.columns if c != "label"]
    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42)
    model.fit(feats[cols].values, feats["label"].values)
    return model, cols


# ----------------------------------------------------------------------
# Levels
# ----------------------------------------------------------------------

def key_levels(daily: pd.DataFrame) -> dict:
    d = daily.iloc[-1]
    h, l, c = float(d["High"]), float(d["Low"]), float(d["Close"])
    p = (h + l + c) / 3
    return {
        "prev_high": h, "prev_low": l, "prev_close": c,
        "pivot": p,
        "r1": 2 * p - l, "s1": 2 * p - h,
        "r2": p + (h - l), "s2": p - (h - l),
    }


# ----------------------------------------------------------------------
# Backtest: daily-bias-filtered ORB on 15m data
# ----------------------------------------------------------------------

def backtest_orb(intra: pd.DataFrame, preds: pd.DataFrame):
    """For each day: take ORB in the direction of the ML bias, if confident."""
    trades = []
    for day, bars in intra.groupby(intra.index.date):
        ts = pd.Timestamp(day)
        if ts not in preds.index:
            continue
        p_up = preds.loc[ts, "p_up"]
        conf = max(p_up, 1 - p_up)
        if conf < CONF_THRESHOLD:
            continue
        direction = 1 if p_up >= 0.5 else -1
        if len(bars) < 6:
            continue
        orb = bars.iloc[:2]                       # 09:15-09:45 opening range
        or_hi, or_lo = float(orb["High"].max()), float(orb["Low"].min())
        risk = or_hi - or_lo
        if risk <= 0:
            continue
        entry = stop = target = None
        result = None
        for _, bar in bars.iloc[2:].iterrows():
            hi, lo, close = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
            if entry is None:
                if direction == 1 and hi > or_hi:
                    entry, stop, target = or_hi, or_lo, or_hi + TARGET_R * risk
                elif direction == -1 and lo < or_lo:
                    entry, stop, target = or_lo, or_hi, or_lo - TARGET_R * risk
                continue
            if direction == 1:
                if lo <= stop:
                    result = -1.0; break
                if hi >= target:
                    result = TARGET_R; break
            else:
                if hi >= stop:
                    result = -1.0; break
                if lo <= target:
                    result = TARGET_R; break
        if entry is not None and result is None:   # exit at close 15:15
            last = float(bars.iloc[-1]["Close"])
            result = direction * (last - entry) / risk
        if result is not None:
            result -= COST_PCT * entry / risk      # costs in R units
            trades.append({"day": day, "dir": "LONG" if direction == 1 else "SHORT",
                           "conf": conf, "R": result})
    return pd.DataFrame(trades)


def print_backtest(name: str, preds: pd.DataFrame, trades: pd.DataFrame):
    acc = (preds["p_up"].round() == preds["label"]).mean()
    confident = preds[(preds["p_up"] - 0.5).abs() >= CONF_THRESHOLD - 0.5]
    acc_conf = ((confident["p_up"].round() == confident["label"]).mean()
                if len(confident) else float("nan"))
    print(f"\n=== {name} — HONEST walk-forward results (last {len(preds)} sessions) ===")
    print(f"  Daily direction accuracy (all days):        {acc:5.1%}")
    print(f"  Accuracy on confident days only ({len(confident):3d} days): {acc_conf:5.1%}")
    if trades.empty:
        print("  ORB backtest: no trades taken in the 60-day intraday window.")
        return
    wins = (trades["R"] > 0).mean()
    total_r = trades["R"].sum()
    eq = trades["R"].cumsum()
    max_dd = (eq - eq.cummax()).min()
    print(f"  ORB trades taken (60d window):  {len(trades)}")
    print(f"  Win rate:                       {wins:5.1%}")
    print(f"  Average R per trade:            {trades['R'].mean():+.2f}R")
    print(f"  Total:                          {total_r:+.1f}R   "
          f"(risking 1% per trade ≈ {total_r:+.1f}% on capital)")
    print(f"  Worst drawdown:                 {max_dd:+.1f}R")
    print(f"  Last 5 trades: " + "  ".join(
        f"{t.day} {t.dir} {t.R:+.2f}R" for t in trades.tail(5).itertuples()))


# ----------------------------------------------------------------------
# Forecast
# ----------------------------------------------------------------------

def print_forecast(name: str, daily: pd.DataFrame, feats: pd.DataFrame,
                   capital: float, risk_pct: float):
    model, cols = fit_full(feats.iloc[:-1] if len(feats) else feats)

    last_session = daily.index[-1].date()
    today = pd.Timestamp.now(tz="Asia/Kolkata").date()
    session_is_today = last_session == today

    if session_is_today:
        # market open/closed today: gap is real, predict today's session
        row = feats.iloc[-1:][cols]
        session_label = f"today ({last_session})"
        levels = key_levels(daily.iloc[:-1])
    else:
        # predicting the NEXT session; gap unknown -> assume flat open
        row = feats.iloc[-1:][cols].copy()
        row["gap"] = 0.0
        row["dow"] = (pd.Timestamp(last_session) + pd.offsets.BDay(1)).dayofweek
        session_label = "next session (flat-open assumption)"
        levels = key_levels(daily)

    p_up = float(model.predict_proba(row.values)[0, 1])
    conf = max(p_up, 1 - p_up)
    bias = "LONG" if p_up >= 0.5 else "SHORT"
    tradeable = conf >= CONF_THRESHOLD

    atr = float(feats.iloc[-1]["atr_pct"] * levels["prev_close"])
    risk_rupees = capital * risk_pct / 100

    print(f"\n{'=' * 62}\n  {name} — forecast for {session_label}\n{'=' * 62}")
    print(f"  Model bias:      {bias}   (P(up) = {p_up:.1%}, confidence = {conf:.1%})")
    print(f"  Verdict:         {'✅ TRADEABLE' if tradeable else '🚫 NO-TRADE DAY — confidence below '
          + f'{CONF_THRESHOLD:.0%}. Sitting out IS the edge.'}")
    print(f"\n  Key levels (from last completed session):")
    print(f"    Prev High / Low / Close :  {levels['prev_high']:.0f} / "
          f"{levels['prev_low']:.0f} / {levels['prev_close']:.0f}")
    print(f"    Pivot                   :  {levels['pivot']:.0f}")
    print(f"    R1 / R2                 :  {levels['r1']:.0f} / {levels['r2']:.0f}")
    print(f"    S1 / S2                 :  {levels['s1']:.0f} / {levels['s2']:.0f}")
    print(f"    ATR (expected day range):  ~{atr:.0f} pts")
    print(f"\n  The plan (Opening Range Breakout, 09:15–09:45):")
    print(f"    1. Mark the HIGH and LOW of the first 30 minutes. Do nothing before 09:45.")
    if bias == "LONG":
        print(f"    2. Bias is LONG → BUY only on a 15m close ABOVE the opening-range HIGH.")
        print(f"    3. Stop-loss  = opening-range LOW.   Target = entry + {TARGET_R}× (high−low).")
    else:
        print(f"    2. Bias is SHORT → SELL only on a 15m close BELOW the opening-range LOW.")
        print(f"    3. Stop-loss  = opening-range HIGH.  Target = entry − {TARGET_R}× (high−low).")
    print(f"    4. If neither side breaks by ~13:30, or the range is huge (> ATR), skip.")
    print(f"    5. Exit everything by 15:15. Never carry an intraday loss overnight.")
    print(f"\n  Position size (capital ₹{capital:,.0f}, risking {risk_pct}% = ₹{risk_rupees:,.0f}):")
    print(f"    Quantity = ₹{risk_rupees:,.0f} ÷ (opening-range size in points).")
    print(f"    e.g. if the opening range is {max(atr*0.35, 1):.0f} pts → "
          f"max {risk_rupees / max(atr*0.35, 1):.1f} units of index exposure.")
    print(f"    ONE trade per index per day. A red day costs {risk_pct}%, never more.")


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Free-data BankNifty/Nifty50 intraday forecaster")
    ap.add_argument("command", choices=["forecast", "backtest"])
    ap.add_argument("--capital", type=float, default=100000, help="trading capital in ₹")
    ap.add_argument("--risk", type=float, default=1.0, help="%% of capital risked per trade")
    args = ap.parse_args()

    print("Fetching free data from Yahoo Finance…")
    for name, symbol in TICKERS.items():
        daily, intra = fetch(symbol)
        feats = build_features(daily)
        if args.command == "backtest":
            preds, _, _ = walk_forward(feats)
            trades = backtest_orb(intra, preds)
            print_backtest(name, preds, trades)
        else:
            print_forecast(name, daily, feats, args.capital, args.risk)

    if args.command == "backtest":
        print("\n" + "-" * 62)
        print("These are honest, no-lookahead numbers. Anything promising 90%+")
        print("accuracy is either overfit or a scam. Edge = discipline × sizing.")
    print()


if __name__ == "__main__":
    main()
