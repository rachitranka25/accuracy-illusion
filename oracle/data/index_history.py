#!/usr/bin/env python3
"""
Fetch long 5-minute history for any Angel-listed index (used to test whether the
system works better on an index we are not currently trading).

    python3 fetch_index.py SENSEX

Writes data_cache/<NAME>_5m_long.csv in the same shape as the existing files, so
every backtest/engine can read it without changes.
"""
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from oracle.data import angel as angel_data
from oracle import paths

CACHE = paths.CACHE
# token, exchange — from Angel's public scrip master (instrumenttype AMXIDX)
INDEXES = {
    "SENSEX":     ("99919000", "BSE"),
    "BANKEX":     ("99919012", "BSE"),
    "FINNIFTY":   ("99926037", "NSE"),
    "MIDCPNIFTY": ("99926074", "NSE"),
}


def fetch(name, months=18, chunk_days=25):
    token, exch = INDEXES[name]
    feed, msg = angel_data.connect()
    if feed is None:
        print(msg)
        return None
    print(f"{msg}\nfetching {name} ({exch}/{token}) — {months} months of 5-min bars…")

    end = datetime.now()
    start = end - timedelta(days=months * 31)
    frames, cur = [], start
    while cur < end:
        stop = min(cur + timedelta(days=chunk_days), end)
        params = {"exchange": exch, "symboltoken": token,
                  "interval": "FIVE_MINUTE",
                  "fromdate": cur.strftime("%Y-%m-%d 09:15"),
                  "todate": stop.strftime("%Y-%m-%d 15:30")}
        try:
            with feed._lock:
                resp = feed.api.getCandleData(params)
            rows = (resp or {}).get("data") or []
            if rows:
                frames.append(pd.DataFrame(
                    rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"]))
                print(f"  {cur:%Y-%m-%d} → {stop:%Y-%m-%d}: {len(rows):5d} bars")
            else:
                print(f"  {cur:%Y-%m-%d} → {stop:%Y-%m-%d}: none")
        except Exception as exc:
            print(f"  {cur:%Y-%m-%d}: {str(exc)[:70]}")
        cur = stop
        time.sleep(2.5)                      # stay under the API rate limit

    if not frames:
        print("nothing fetched")
        return None
    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"])
    df = (df.drop_duplicates(subset="ts").sort_values("ts")
            .set_index("ts").astype(float))
    df.index = df.index.tz_localize(None) if df.index.tz is not None else df.index
    out = CACHE / f"{name}_5m_long.csv"
    df.to_csv(out)
    days = len(set(df.index.date))
    print(f"saved {out.name}: {len(df)} bars over {days} sessions "
          f"({df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d})")
    return df


if __name__ == "__main__":
    name = (sys.argv[1] if len(sys.argv) > 1 else "SENSEX").upper()
    if name not in INDEXES:
        print(f"unknown index. known: {', '.join(INDEXES)}")
        sys.exit(1)
    fetch(name)
