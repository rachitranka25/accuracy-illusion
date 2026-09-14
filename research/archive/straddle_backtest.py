#!/usr/bin/env python3
"""
Short-Straddle backtest — clean, COST-AWARE template.

Strategy (classic intraday short straddle):
  - At ENTRY_TIME (default 09:20): SELL the ATM Call (CE) and ATM Put (PE).
  - Per-leg stop-loss: if a leg's premium RISES by SL_PCT above entry, buy it
    back (a loss on that leg).
  - At EXIT_TIME (default 15:10): buy back any leg still open.
  - Profit comes from premium DECAY (theta) when the index stays calm.

Why this file exists: the public GitHub straddle backtests omit costs, which
makes them look profitable when they are not. This version subtracts realistic
BROKERAGE + SLIPPAGE on every fill — the single most important honesty fix.

REALITY CHECK: short straddles collect a small premium most days and take an
occasional LARGE loss on a big-move day (the SL caps it but whipsaws). Only a
proper cost-aware backtest over many expiries tells you if the edge is real.
There is no guaranteed edge here — this is a template to TEST, not a promise.

INPUT DATA FORMAT (once real option data arrives, e.g. from Angel/friend):
  a DataFrame with one row per intraday timestamp, columns:
    datetime  (pandas Timestamp, IST)
    future    (BankNifty/Nifty futures or index price)
    ce        (ATM Call premium at that time)
    pe        (ATM Put premium at that time)
  The ATM strike is chosen at entry; ce/pe should track that same strike's
  premium through the day. One straddle per day.

Run now (synthetic demo, just to verify the engine):  python3 straddle_backtest.py
"""
import argparse
import numpy as np
import pandas as pd
from datetime import time as dtime

# ---- cost model (edit to your broker) -----------------------------------
LOT_SIZE = 15            # BankNifty lot (verify current with your broker)
BROKERAGE_PER_ORDER = 20  # flat Rs/order (a straddle = 4 orders: 2 legs x in+out)
SLIPPAGE_PCT = 0.01      # 1% of premium lost to spread/slippage per fill
SL_GAP_SLIP = 0.10       # SL fills 10% WORSE than trigger (gap-through on fast moves)
ENTRY_TIME = dtime(9, 20)
EXIT_TIME = dtime(15, 10)
SL_PCT = 0.20            # 20% per-leg stop-loss


def _fill_cost(premium, qty):
    """Brokerage + slippage for ONE fill of `qty` units at `premium`."""
    return BROKERAGE_PER_ORDER + SLIPPAGE_PCT * premium * qty


def backtest_day(day_df, lot=LOT_SIZE, sl_pct=SL_PCT):
    """Run one day's short straddle. Returns a dict, or None if data is thin."""
    d = day_df.sort_values("datetime").reset_index(drop=True)
    entry = d[d["datetime"].dt.time >= ENTRY_TIME]
    if entry.empty:
        return None
    e = entry.iloc[0]
    ce_in, pe_in = float(e["ce"]), float(e["pe"])
    if ce_in <= 0 or pe_in <= 0:
        return None
    ce_sl, pe_sl = ce_in * (1 + sl_pct), pe_in * (1 + sl_pct)
    qty = lot

    ce_out = pe_out = None                 # exit premiums (None = still open)
    for _, row in d[d["datetime"] >= e["datetime"]].iterrows():
        t = row["datetime"].time()
        if ce_out is None and float(row["ce"]) >= ce_sl:
            ce_out = ce_sl * (1 + SL_GAP_SLIP)   # stopped out, worse fill on gap
        if pe_out is None and float(row["pe"]) >= pe_sl:
            pe_out = pe_sl * (1 + SL_GAP_SLIP)
        if t >= EXIT_TIME:
            if ce_out is None:
                ce_out = float(row["ce"])
            if pe_out is None:
                pe_out = float(row["pe"])
            break
    if ce_out is None:
        ce_out = float(d.iloc[-1]["ce"])
    if pe_out is None:
        pe_out = float(d.iloc[-1]["pe"])

    # seller PnL = (premium collected - premium bought back) * qty
    gross = ((ce_in - ce_out) + (pe_in - pe_out)) * qty
    costs = (_fill_cost(ce_in, qty) + _fill_cost(ce_out, qty)
             + _fill_cost(pe_in, qty) + _fill_cost(pe_out, qty))
    return {"date": e["datetime"].date(), "ce_in": ce_in, "pe_in": pe_in,
            "ce_out": ce_out, "pe_out": pe_out, "gross": gross,
            "costs": costs, "net": gross - costs}


def backtest(df, lot=LOT_SIZE, sl_pct=SL_PCT):
    """Run over all days in df. Returns (trades_df, summary_dict)."""
    rows = []
    for _, day_df in df.groupby(df["datetime"].dt.date):
        r = backtest_day(day_df, lot, sl_pct)
        if r:
            rows.append(r)
    t = pd.DataFrame(rows)
    if t.empty:
        return t, {}
    net = t["net"]
    eq = net.cumsum()
    summary = {
        "days": len(t),
        "win_rate": round((net > 0).mean() * 100, 1),
        "avg_net": round(net.mean()),
        "total_net": round(net.sum()),
        "total_gross": round(t["gross"].sum()),
        "total_costs": round(t["costs"].sum()),
        "best_day": round(net.max()),
        "worst_day": round(net.min()),
        "max_drawdown": round((eq - eq.cummax()).min()),
    }
    return t, summary


# ---- synthetic demo so the engine is verifiable NOW ---------------------
def _demo_data(days=40, seed=1):
    """Fake intraday ATM CE/PE paths: premium decays through the day + moves.
    NOT real — only to prove the backtest engine runs correctly."""
    rng = np.random.default_rng(seed)
    out = []
    base = pd.Timestamp("2026-06-01 09:15")
    for dnum in range(days):
        day = base + pd.Timedelta(days=dnum)
        if day.dayofweek >= 5:
            continue
        fut = 50000 + rng.normal(0, 300)
        # most days calm; ~12% trend days; ~10% whipsaw (both legs spike = double loss)
        move = rng.normal(0, 1)
        whipsaw = rng.random() < 0.10
        if not whipsaw and rng.random() < 0.13:
            move = rng.choice([-1, 1]) * rng.uniform(3.5, 6.0)   # trend/gap day
        ce0 = pe0 = 250 + rng.normal(0, 30)         # ATM premium ~250
        times = pd.date_range(day.replace(hour=9, minute=15),
                              day.replace(hour=15, minute=25), freq="5min")
        for i, ts in enumerate(times):
            frac = i / len(times)
            decay = (1 - 0.25 * frac)               # theta: modest intraday bleed
            drift = move * frac * 90                 # directional move hurts a leg
            spike = (ce0 * 0.55 * min(1, frac * 2)) if whipsaw else 0  # vol spike both legs
            ce = max(2, ce0 * decay + drift + spike + rng.normal(0, 10))
            pe = max(2, pe0 * decay - drift + spike + rng.normal(0, 10))
            out.append({"datetime": ts, "future": fut, "ce": ce, "pe": pe})
    return pd.DataFrame(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="real option data CSV (datetime,future,ce,pe)")
    ap.add_argument("--lot", type=int, default=LOT_SIZE)
    ap.add_argument("--sl", type=float, default=SL_PCT)
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv, parse_dates=["datetime"])
        print(f"Loaded {len(df)} rows from {args.csv}")
    else:
        df = _demo_data()
        print("=== DEMO MODE (synthetic data — numbers are NOT real) ===")

    trades, s = backtest(df, args.lot, args.sl)
    if not s:
        print("no trades"); raise SystemExit
    print(f"\n  days traded     : {s['days']}")
    print(f"  win rate        : {s['win_rate']}%")
    print(f"  avg net / day   : Rs {s['avg_net']:,}")
    print(f"  TOTAL net       : Rs {s['total_net']:,}   "
          f"(gross Rs {s['total_gross']:,} - costs Rs {s['total_costs']:,})")
    print(f"  best / worst day: Rs {s['best_day']:,} / Rs {s['worst_day']:,}")
    print(f"  max drawdown    : Rs {s['max_drawdown']:,}")
    print(f"\n  costs ate {s['total_costs']/max(abs(s['total_gross']),1)*100:.0f}% "
          f"of gross — this is why cost-free backtests lie.")
    print("  Replace demo with REAL option data (--csv) to get a real verdict.")
