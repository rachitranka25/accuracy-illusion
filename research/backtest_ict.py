#!/usr/bin/env python3
"""
Honest mechanical backtest of ICT's core "liquidity sweep reversal" setup
(Mullham Trading's favourite Trade-1), adapted to NSE intraday.

NO hindsight, NO discretion. Fixed rules:

  1. Opening Range (OR) = High/Low of the first `OR_MIN` minutes (9:15 -> ...).
     This is the "session liquidity" that gets built, analogous to his Asian range.
  2. After the OR window, on each 5-min bar look for a SWEEP that FAILS:
       - bar HIGH > OR_high  but  bar CLOSE < OR_high  -> stop-hunt up  -> SHORT
       - bar LOW  < OR_low   but  bar CLOSE > OR_low   -> stop-hunt down -> LONG
     (this is "take liquidity then reject" — his weakness/turtle-soup signature)
  3. Entry = that bar's close. ONE trade/day (first valid sweep only).
  4. Stop = beyond the sweep bar's extreme + buffer.
  5. Target = opposite side of the OR (internal->external liquidity).
  6. Also exit at session close if neither hit.
  7. Realistic round-trip cost applied in points.

We report: direction hit-rate, target-hit rate, avg R, expectancy (R/trade),
and compare to a coin-flip baseline. Honest numbers only.
"""
import sys
import pandas as pd
import numpy as np

from oracle import paths

OR_MIN = 45           # opening-range window in minutes (9:15-10:00)
BUF_BP = 0.0003       # stop buffer beyond sweep extreme (3 bps)
COST_BP = 0.0004      # round-trip cost in fraction of price (~4 bps, futures+slip)
SESSION_END = "15:15" # square off
TRADE_AFTER = "10:00" # only look for sweeps after OR window closes


def load(path):
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"])
    df["date"] = df["ts"].dt.date
    df["hm"] = df["ts"].dt.strftime("%H:%M")
    return df


def backtest(df, name):
    trades = []
    for d, g in df.groupby("date"):
        g = g.sort_values("ts").reset_index(drop=True)
        oro = g[g["hm"] <= "10:00"]
        if len(oro) < 5:
            continue
        or_hi = oro["High"].max()
        or_lo = oro["Low"].min()
        rng = or_hi - or_lo
        if rng <= 0:
            continue
        after = g[(g["hm"] > TRADE_AFTER) & (g["hm"] <= SESSION_END)].reset_index(drop=True)
        if len(after) < 3:
            continue
        entry = None
        for i, r in after.iterrows():
            if r["High"] > or_hi and r["Close"] < or_hi:      # failed sweep up -> short
                entry = dict(side=-1, px=r["Close"], stop=r["High"] * (1 + BUF_BP),
                             tgt=or_lo, i=i); break
            if r["Low"] < or_lo and r["Close"] > or_lo:       # failed sweep down -> long
                entry = dict(side=1, px=r["Close"], stop=r["Low"] * (1 - BUF_BP),
                             tgt=or_hi, i=i); break
        if not entry:
            continue
        side, px, stop, tgt = entry["side"], entry["px"], entry["stop"], entry["tgt"]
        risk = abs(px - stop)
        if risk <= 0:
            continue
        # walk forward bar by bar; stop and target checked intrabar (stop first = conservative)
        outcome, exit_px = None, None
        for j in range(entry["i"] + 1, len(after)):
            b = after.iloc[j]
            if side == 1:
                if b["Low"] <= stop:  outcome, exit_px = "stop", stop; break
                if b["High"] >= tgt:  outcome, exit_px = "target", tgt; break
            else:
                if b["High"] >= stop: outcome, exit_px = "stop", stop; break
                if b["Low"] <= tgt:   outcome, exit_px = "target", tgt; break
        if outcome is None:            # timed out -> exit at last close
            exit_px = after.iloc[-1]["Close"]; outcome = "timeout"
        gross = side * (exit_px - px)
        cost = px * COST_BP
        net = gross - cost
        trades.append(dict(date=d, side=side, entry=px, exitp=exit_px,
                           risk=risk, R=net / risk, net=net, outcome=outcome))
    t = pd.DataFrame(trades)
    print(f"\n===== {name} =====")
    if t.empty:
        print("no trades"); return
    n = len(t)
    wins = (t["net"] > 0).sum()
    tgt = (t["outcome"] == "target").sum()
    stp = (t["outcome"] == "stop").sum()
    tmo = (t["outcome"] == "timeout").sum()
    exp = t["R"].mean()
    print(f"trades          : {n}  ({t['date'].min()} -> {t['date'].max()})")
    print(f"net win-rate    : {wins/n*100:4.1f}%   (coin-flip ~50%)")
    print(f"  target hit    : {tgt/n*100:4.1f}%   stop {stp/n*100:4.1f}%   timeout {tmo/n*100:4.1f}%")
    print(f"expectancy      : {exp:+.3f} R/trade   (needs > 0 after cost to be real edge)")
    print(f"avg R win       : {t.loc[t.net>0,'R'].mean():+.2f}   avg R loss {t.loc[t.net<=0,'R'].mean():+.2f}")
    print(f"total R         : {t['R'].sum():+.1f} over {n} trades")
    # equity curve sanity
    eq = t["R"].cumsum()
    print(f"max drawdown R  : {(eq.cummax()-eq).max():.1f}")
    return t


if __name__ == "__main__":
    for nm, f in [("BANKNIFTY", paths.CACHE / "BANKNIFTY_5m_long.csv"),
                  ("NIFTY50",   paths.CACHE / "NIFTY50_5m_long.csv")]:
        try:
            backtest(load(f), nm)
        except Exception as e:
            print(nm, "error:", e)
    print("\nNote: expectancy (R/trade) is what matters, not win-rate. A 45% win-rate "
          "with >1.5 avg-win/avg-loss is profitable; a 60% win-rate with tiny wins is not.")
