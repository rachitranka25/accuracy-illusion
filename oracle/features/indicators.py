#!/usr/bin/env python3
"""
NIFTY LIVE — minute-by-minute signal engine for BankNifty & Nifty50.

Every 60 seconds it prints one line per index telling you EXACTLY what to
do right now — WAIT, or:

  🔔 BUY 3.4 units @ 57420 | SL 57250 (-170) | TGT 57675 (+255)
     | hold until TGT/SL or 15:15 | confidence 62%

and then manages the position minute by minute until stop, target or the
15:15 square-off.

Strategy (the ONLY variant that survived honest backtesting):
  - A daily ML model (2 years of data) sets the day's bias at the open.
  - 09:15–09:45: build the opening range. No trading.
  - After 09:45: enter on a 1-minute close beyond the opening range in
    the bias direction. Stop = other side of the range. Target = 1.5R.
  - ONE trade per index per day. No entries after 13:30. Flat by 15:15.

I also tested a rapid-fire ML scalper (many trades/day on 5m bars):
it LOST money in all 18 tested configurations, so it was removed.
That is what honest engineering looks like.

Usage
-----
  python3 nifty_live.py live     [--capital 100000] [--risk 1.0]  # during market hours
  python3 nifty_live.py replay   [--capital 100000] [--risk 1.0]  # demo on the last session
  python3 nifty_live.py backtest                                  # honest walk-forward stats
"""

import argparse
import sys
import time as _time
import warnings
from datetime import time as dtime

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

TICKERS = {"BANKNIFTY": "^NSEBANK", "NIFTY50": "^NSEI",
           "FINNIFTY": "NIFTY_FIN_SERVICE.NS"}
TZ = "Asia/Kolkata"

CONF_THRESHOLD = 0.58        # below this the whole day is NO-TRADE
TARGET_R = 1.5               # target = 1.5x risk
COST_PCT = 0.0004            # slippage + charges per round trip
MAX_LAG_MIN = 5              # refuse to signal on stale data
OR_END = dtime(9, 45)        # opening range = 09:15 - 09:45
LAST_ENTRY = dtime(13, 30)
SQUARE_OFF = dtime(15, 15)


def flat(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# ----------------------------------------------------------------------
# Daily bias model (same as nifty_oracle.py)
# ----------------------------------------------------------------------

def build_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
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
    # --- indicators ---
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["macd_hist"] = ((macd - macd.ewm(span=9).mean()) / c).shift(1)
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = ((c - ma20) / (2 * sd20)).shift(1)          # Bollinger %b
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    f["stoch"] = ((c - lo14) / (hi14 - lo14)).shift(1)        # stochastic %K
    # --- candlestick patterns (previous completed day) ---
    body, rng = c - o, (h - l).replace(0, np.nan)
    f["doji"] = (body.abs() / rng < 0.1).astype(float).shift(1)
    f["engulf_bull"] = ((body > 0) & (body.shift(1) < 0) &
                        (c > o.shift(1)) & (o < c.shift(1))).astype(float).shift(1)
    f["engulf_bear"] = ((body < 0) & (body.shift(1) > 0) &
                        (c < o.shift(1)) & (o > c.shift(1))).astype(float).shift(1)
    f["hammer"] = (((np.minimum(o, c) - l) > 2 * body.abs()) &
                   ((h - np.maximum(o, c)) < body.abs())).astype(float).shift(1)
    f["gap"] = o / c.shift(1) - 1        # known at 09:15
    f["dow"] = f.index.dayofweek
    f["label"] = (c > o).astype(int)
    return f.dropna()


def make_model():
    return GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=42)


BASE_COLS = ["ret1", "ret3", "ret5", "rsi", "ema20_dist", "ema50_dist",
             "atr_pct", "range_pos", "gap", "dow"]
INDICATOR_COLS = ["macd_hist", "bb_pos", "stoch"]
PATTERN_COLS = ["doji", "engulf_bull", "engulf_bear", "hammer"]
DAILY_COLS = BASE_COLS + INDICATOR_COLS + PATTERN_COLS

# Chosen by walk-forward backtest: indicator features improve BANKNIFTY
# (+2.2R -> +3.0R) but overfit NIFTY50 (+2.7R -> -0.3R); pattern flags
# helped neither, so patterns stay UI-only.
FEATURE_SETS = {"BANKNIFTY": BASE_COLS + INDICATOR_COLS, "NIFTY50": BASE_COLS}


def walk_forward_bias(feats: pd.DataFrame, test_days: int = 150,
                      cols=None) -> pd.DataFrame:
    """Expanding-window daily predictions — no lookahead."""
    cols = cols or DAILY_COLS
    X, y = feats[cols].values, feats["label"].values
    n = len(feats)
    start = max(n - test_days, 60)
    rows, model = [], None
    for i in range(start, n):
        if model is None or (i - start) % 10 == 0:
            model = make_model()
            model.fit(X[:i], y[:i])
        rows.append((feats.index[i].date(), model.predict_proba(X[i:i + 1])[0, 1]))
    return pd.DataFrame(rows, columns=["date", "p_up"]).set_index("date")


def bias_for_day(feats: pd.DataFrame, day, cols=None) -> float:
    """P(up) for `day`, model trained strictly on earlier days."""
    cols = cols or DAILY_COLS
    train = feats[feats.index.date < day]
    model = make_model()
    model.fit(train[cols].values, train["label"].values)
    row = feats[feats.index.date == day]
    if row.empty:
        return float("nan")
    return float(model.predict_proba(row[cols].values)[0, 1])


# ----------------------------------------------------------------------
# ORB engine — consumes one minute-bar at a time, emits instruction lines
# ----------------------------------------------------------------------

class ORBEngine:
    def __init__(self, name, p_up, capital, risk_pct):
        self.name = name
        self.p_up = p_up
        self.conf = max(p_up, 1 - p_up)
        self.dir = 1 if p_up >= 0.5 else -1
        self.tradeable = self.conf >= CONF_THRESHOLD
        self.adaptive = False     # BANKNIFTY only: unlock on 4/4 indicator alignment
        self.aligned = 0          # caller sets this each bar (0-4 indicators agreeing)
        self.risk_rupees = capital * risk_pct / 100
        self.or_hi = self.or_lo = None
        self.pos = None
        self.done = False
        self.result_r = None
        self.results = []          # R of every closed trade today
        self.trade_count = 0
        self.max_trades = 3        # backtested: 3/day with re-entry beats 1/day
        self.last_entry = LAST_ENTRY   # per-index; BANKNIFTY extends to 14:30

    def bias_str(self):
        side = "LONG" if self.dir == 1 else "SHORT"
        return f"bias {side} {self.conf:.0%}"

    def on_minute(self, ts, o, h, l, c):
        t = ts.time()
        n = self.name.ljust(9)

        if t < OR_END:                                # building the range
            self.or_hi = h if self.or_hi is None else max(self.or_hi, h)
            self.or_lo = l if self.or_lo is None else min(self.or_lo, l)
            return (f"{n} ▶ WAIT — building opening range "
                    f"[{self.or_lo:.0f} … {self.or_hi:.0f}] | {self.bias_str()}")

        if self.pos:                                  # manage open trade
            p = self.pos
            r = None
            if p["dir"] == 1 and l <= p["sl"]:
                r, tag = -1.0, "🛑 STOP HIT"
            elif p["dir"] == 1 and h >= p["tgt"]:
                r, tag = TARGET_R, "🎯 TARGET HIT"
            elif p["dir"] == -1 and h >= p["sl"]:
                r, tag = -1.0, "🛑 STOP HIT"
            elif p["dir"] == -1 and l <= p["tgt"]:
                r, tag = TARGET_R, "🎯 TARGET HIT"
            elif t >= SQUARE_OFF:
                r = p["dir"] * (c - p["entry"]) / p["risk_pts"]
                tag = "⏰ 15:15 SQUARE-OFF"
            if r is not None:
                r -= COST_PCT * p["entry"] / p["risk_pts"]
                self.result_r = r
                self.results.append(r)
                self.trade_count += 1
                self.pos = None
                if self.trade_count >= self.max_trades or t >= SQUARE_OFF:
                    self.done = True
                side = "LONG" if p["dir"] == 1 else "SHORT"
                tail = (" Done for today." if self.done else
                        f" Trade {self.trade_count}/{self.max_trades} — "
                        f"watching for the next breakout.")
                return (f"{n} ▶ {tag}: exit {side} → {r:+.2f}R "
                        f"(₹{r * self.risk_rupees:+,.0f}).{tail}")
            side = "LONG " if p["dir"] == 1 else "SHORT"
            upnl = p["dir"] * (c - p["entry"]) * p["qty"]
            return (f"{n} ▶ HOLD {side} {p['qty']:.1f}u @ {p['entry']:.0f} | "
                    f"now {c:.0f} (₹{upnl:+,.0f}) | SL {p['sl']:.0f} | TGT {p['tgt']:.0f} "
                    f"| exit by 15:15")

        if self.done:
            r = (f"{len(self.results)} trade(s), total {sum(self.results):+.2f}R"
                 if self.results else "no trade")
            return f"{n} ▶ DONE for today ({r}). Discipline over adrenaline."
        allowed = self.tradeable or (self.adaptive and self.aligned >= 4)
        if not self.tradeable and not self.adaptive:
            return (f"{n} ▶ 🚫 NO-TRADE DAY (confidence {self.conf:.0%} < "
                    f"{CONF_THRESHOLD:.0%}). Sitting out IS the trade.")
        if t > self.last_entry:
            self.done = True
            le = self.last_entry.strftime("%H:%M")
            if self.results:
                return (f"{n} ▶ Entry window closed ({le}). "
                        f"{len(self.results)} trade(s), total {sum(self.results):+.2f}R.")
            return f"{n} ▶ No breakout by {le} — day skipped. That's discipline, not failure."

        risk_pts = self.or_hi - self.or_lo
        if risk_pts <= 0:
            return f"{n} ▶ WAIT (range invalid)"
        breakout = (self.dir == 1 and c > self.or_hi) or \
                   (self.dir == -1 and c < self.or_lo)
        if breakout and not allowed:
            # adaptive day: breakout came but indicators disagreed -> skip (backtested)
            self.done = True
            return (f"{n} ▶ Breakout came but only {self.aligned}/4 indicators "
                    f"agreed (needs 4/4 on low-confidence days) — day skipped.")
        if not allowed:
            need = "bearish" if self.dir == -1 else "bullish"
            return (f"{n} ▶ WATCH | conf {self.conf:.0%} < {CONF_THRESHOLD:.0%} | "
                    f"unlocks only if 4/4 {need} indicators at breakout "
                    f"(now {self.aligned}/4 {need})")
        if self.dir == 1 and c > self.or_hi:
            entry, sl = c, self.or_lo
        elif self.dir == -1 and c < self.or_lo:
            entry, sl = c, self.or_hi
        else:
            trig = self.or_hi if self.dir == 1 else self.or_lo
            word = "ABOVE" if self.dir == 1 else "BELOW"
            return (f"{n} ▶ WAIT | {self.bias_str()} | price {c:.0f} | "
                    f"enter on 1m close {word} {trig:.0f}")
        risk_pts = abs(entry - sl)
        tgt = entry + self.dir * TARGET_R * risk_pts
        qty = self.risk_rupees / risk_pts
        self.pos = {"dir": self.dir, "entry": entry, "sl": sl, "tgt": tgt,
                    "qty": qty, "risk_pts": risk_pts}
        act = "BUY " if self.dir == 1 else "SELL"
        return (f"{n} ▶ 🔔 {act} {qty:.1f} units @ {entry:.0f} | "
                f"SL {sl:.0f} ({-self.dir * risk_pts:+.0f}) | "
                f"TGT {tgt:.0f} ({self.dir * TARGET_R * risk_pts:+.0f}) | "
                f"hold until TGT/SL or 15:15 | confidence {self.conf:.0%} "
                f"| max loss ₹{self.risk_rupees:,.0f}")


# ----------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------

def fetch_daily(symbol):
    return flat(yf.download(symbol, period="2y", interval="1d",
                            progress=False, auto_adjust=True))


def run_replay(capital, risk_pct):
    for name, symbol in TICKERS.items():
        daily = fetch_daily(symbol)
        m1 = flat(yf.download(symbol, period="5d", interval="1m",
                              progress=False, auto_adjust=True))
        day = sorted(set(m1.index.date))[-1]
        feats = build_daily_features(daily)
        p_up = bias_for_day(feats, day, FEATURE_SETS.get(name))
        if np.isnan(p_up):
            print(f"{name}: no daily row for {day}, skipping"); continue
        eng = ORBEngine(name, p_up, capital, risk_pct)
        bars = m1[m1.index.date == day]
        print(f"\n===== {name} — replay of {day} "
              f"(model trained only on earlier days) =====")
        prev = None
        for ts, bar in bars.iterrows():
            line = eng.on_minute(ts, float(bar["Open"]), float(bar["High"]),
                                 float(bar["Low"]), float(bar["Close"]))
            key = line.split("|")[0]
            interesting = ("🔔" in line or "HIT" in line or "SQUARE" in line
                           or "skipped" in line or key != prev or ts.minute % 15 == 0)
            if interesting:
                print(f"  {ts:%H:%M}  {line}")
            prev = key
        if eng.result_r is not None:
            print(f"  RESULT: {eng.result_r:+.2f}R  "
                  f"(₹{eng.result_r * eng.risk_rupees:+,.0f} at your risk setting)")
        else:
            print("  RESULT: no trade taken (by design).")


def run_backtest(capital, risk_pct):
    print("Walk-forward backtest of the EXACT live rules on 60 days of "
          "5-minute data.\nNo lookahead, costs included.\n")
    for name, symbol in TICKERS.items():
        daily = fetch_daily(symbol)
        m5 = flat(yf.download(symbol, period="60d", interval="5m",
                              progress=False, auto_adjust=True))
        feats = build_daily_features(daily)
        preds = walk_forward_bias(feats, cols=FEATURE_SETS.get(name))
        results = []
        for day, bars in m5.groupby(m5.index.date):
            if day not in preds.index:
                continue
            eng = ORBEngine(name, float(preds.loc[day, "p_up"]), capital, risk_pct)
            for ts, bar in bars.iterrows():
                eng.on_minute(ts, float(bar["Open"]), float(bar["High"]),
                              float(bar["Low"]), float(bar["Close"]))
            if eng.pos:                       # square off at last bar
                last = float(bars.iloc[-1]["Close"])
                p = eng.pos
                r = p["dir"] * (last - p["entry"]) / p["risk_pts"] - \
                    COST_PCT * p["entry"] / p["risk_pts"]
                eng.result_r = r
            if eng.result_r is not None:
                results.append(eng.result_r)
        rs = pd.Series(results)
        print(f"=== {name} ===")
        if rs.empty:
            print("  No trades.\n"); continue
        eq = rs.cumsum()
        print(f"  Trades:          {len(rs)} over ~60 sessions "
              f"(≈{len(rs) / 60 * 5:.1f}/week — patience is the strategy)")
        print(f"  Win rate:        {(rs > 0).mean():.1%}")
        print(f"  Avg R:           {rs.mean():+.2f}R")
        print(f"  Total:           {rs.sum():+.1f}R  "
              f"(≈ {rs.sum() * risk_pct:+.1f}% on capital at {risk_pct}% risk/trade)")
        print(f"  Worst drawdown:  {(eq - eq.cummax()).min():+.1f}R\n")
    print("-" * 62)
    print("Modest, real numbers. The rapid-fire scalper version lost money in")
    print("all 18 tested configs and was removed. Never trust 90%+ claims.")


def run_live(capital, risk_pct):
    now = pd.Timestamp.now(tz=TZ)
    if now.dayofweek >= 5 or not (dtime(9, 15) <= now.time() <= dtime(15, 30)):
        print(f"Market is closed (Mon–Fri 09:15–15:30 IST; now {now:%a %H:%M}).")
        print("Run  python3 nifty_live.py replay  to see a full demo of the last session.")
        return
    print("Training daily bias models on 2 years of data…")
    engines = {}
    for name, symbol in TICKERS.items():
        daily = fetch_daily(symbol)
        feats = build_daily_features(daily)
        cols = FEATURE_SETS.get(name, DAILY_COLS)
        p_up = bias_for_day(feats, now.date(), cols)
        if np.isnan(p_up):                      # today's daily bar not up yet
            model = make_model()
            model.fit(feats[cols].values, feats["label"].values)
            row = feats.iloc[-1:][cols].copy()
            row["gap"] = 0.0
            row["dow"] = now.dayofweek
            p_up = float(model.predict_proba(row.values)[0, 1])
        eng = ORBEngine(name, p_up, capital, risk_pct)
        engines[name] = (symbol, eng)
        print(f"  {name}: {eng.bias_str()} → "
              f"{'tradeable ✅' if eng.tradeable else 'NO-TRADE day 🚫'}")
    print("\nLive. One line per index every 60 seconds. Ctrl+C to stop.\n")
    seen = {name: None for name in engines}
    try:
        while True:
            now = pd.Timestamp.now(tz=TZ)
            if now.time() > dtime(15, 30):
                print("Market closed. Session over."); break
            stamp = now.strftime("%H:%M:%S")
            for name, (symbol, eng) in engines.items():
                try:
                    m1 = flat(yf.download(symbol, period="1d", interval="1m",
                                          progress=False, auto_adjust=True))
                except Exception as exc:
                    print(f"[{stamp}] {name}: fetch failed ({exc}); retrying"); continue
                m1 = m1[m1.index.date == now.date()]
                if m1.empty:
                    print(f"[{stamp}] {name}: waiting for today's data…"); continue
                lag = (now - m1.index[-1]).total_seconds() / 60
                if lag > MAX_LAG_MIN:
                    print(f"[{stamp}] {name:9s} ▶ ⚠️ data {lag:.0f} min stale — "
                          f"not acting on old prices"); continue
                # feed any minute bars we haven't processed yet, in order
                new = m1 if seen[name] is None else m1[m1.index > seen[name]]
                line = None
                for ts, bar in new.iterrows():
                    line = eng.on_minute(ts, float(bar["Open"]), float(bar["High"]),
                                         float(bar["Low"]), float(bar["Close"]))
                seen[name] = m1.index[-1]
                if line:
                    print(f"[{stamp}] {line} | lag {lag:.0f}m")
            sys.stdout.flush()
            _time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    for name, (_, eng) in engines.items():
        if eng.result_r is not None:
            print(f"  {name}: {eng.result_r:+.2f}R "
                  f"(₹{eng.result_r * eng.risk_rupees:+,.0f})")
        elif eng.pos:
            print(f"  {name}: ⚠️ position still open — close it manually in your broker!")


def main():
    ap = argparse.ArgumentParser(description="Minute-by-minute BankNifty/Nifty50 signals")
    ap.add_argument("command", choices=["live", "replay", "backtest"])
    ap.add_argument("--capital", type=float, default=100000)
    ap.add_argument("--risk", type=float, default=1.0, help="%% of capital risked per trade")
    args = ap.parse_args()
    if args.command == "live":
        run_live(args.capital, args.risk)
    elif args.command == "replay":
        run_replay(args.capital, args.risk)
    else:
        run_backtest(args.capital, args.risk)


if __name__ == "__main__":
    main()
