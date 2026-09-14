#!/usr/bin/env python3
"""
Smart-money daily bias — from NSE participant-wise Open Interest (FII positioning).

Signal (contrarian, tested 2yr out-of-sample, per-day, stable both halves):
  When FII's net index-futures position rises ABOVE its 20-day average
  (FII more bullish than usual) -> next day tends DOWN.  Below average -> UP.
  i.e. FADE extreme FII positioning.  BankNifty ~60% next-day, Nifty ~55%,
  positive after costs. This is POSITIONAL (next-day), not intraday.

Data comes from data_cache/participant_oi.csv (fetch_participant_oi.py, daily free).
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
from oracle import paths

CACHE = paths.CACHE
# measured next-day direction accuracy of the contrarian rule (2yr OOS)
ACC = {"BANKNIFTY": 60, "NIFTY50": 55}


def _load():
    f = CACHE / "participant_oi.csv"
    if not f.exists():
        return None
    p = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
    for c in p.columns:
        if c != "date":
            p[c] = pd.to_numeric(p[c], errors="coerce")
    if "FII_fut_long" not in p or "FII_fut_short" not in p:
        return None
    p["FII_net"] = p["FII_fut_long"] - p["FII_fut_short"]
    return p.set_index("date")


def bias():
    """Return {index: {'lean':'LONG'|'SHORT', 'acc':pct, 'as_of':date,
    'fii_net':int, 'dev':float}} using the latest available OI day."""
    p = _load()
    if p is None or len(p) < 21:
        return {}
    dev = (p["FII_net"] - p["FII_net"].rolling(20).mean()).dropna()
    if dev.empty:
        return {}
    last = dev.index[-1]
    d = float(dev.iloc[-1])
    # contrarian: FII above its avg (positive dev) -> SHORT ; below -> LONG
    lean = "SHORT" if d > 0 else "LONG"
    out = {}
    for idx in ("BANKNIFTY", "NIFTY50"):
        out[idx] = {"lean": lean, "acc": ACC[idx],
                    "as_of": last.strftime("%Y-%m-%d"),
                    "fii_net": int(p["FII_net"].iloc[-1]), "dev": round(d)}
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(bias(), indent=2))
