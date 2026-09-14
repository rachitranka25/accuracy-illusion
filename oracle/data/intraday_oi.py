"""
Intraday open interest at 10-minute resolution.

The EOD bhavcopy gives one OI reading per day, and that reading turned out to be
a slow-moving proxy for price -- 0.818 correlated with the 10-day return, which
is why it added nothing. Intraday OI is a different animal: it says whether
positions are being built or unwound *while the session is running*, which is
information the price series does not carry.

The source keeps a rolling window of roughly a month, so this appends to a local
file rather than overwriting. Run it daily and the history grows past what the
site itself will serve.
"""
import json
import subprocess
import datetime as dt
from pathlib import Path

import pandas as pd
from oracle import paths

CACHE = paths.CACHE
BACKEND = "https://tradingtick.in/nifty/php-files/all-live-pcr-backend.php"
REFERER = "https://tradingtick.in/nifty/nifty-banknifty-finnifty-intraday-pcr-live.php"
INDEXES = ["BANKNIFTY", "NIFTY", "FINNIFTY"]


def _get(index, day):
    url = f"{BACKEND}?action=getChartData&index={index}&date={day}"
    out = subprocess.run(
        ["curl", "-s", "-A", "Mozilla/5.0", "-H", "X-Requested-With: XMLHttpRequest",
         "-e", REFERER, "--max-time", "20", url],
        capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except Exception:
        return {}


def fetch_day(index, day):
    j = _get(index, day)
    times, ce, pe = j.get("times", []), j.get("ceOi", []), j.get("peOi", [])
    if not times:
        return pd.DataFrame()
    rows = []
    for t, c, p in zip(times, ce, pe):
        try:
            c, p = float(c), float(p)
        except (TypeError, ValueError):
            continue
        # The feed writes 0 when a snapshot is missing rather than omitting the
        # point. Zero OI is impossible on an index chain, so those are dropped
        # instead of being carried into the features as a real collapse.
        if c <= 0 or p <= 0:
            continue
        rows.append({"ts": pd.Timestamp(f"{day} {t}"), "ce_oi": c, "pe_oi": p})
    return pd.DataFrame(rows)


def update(index, back=45):
    path = CACHE / f"{index}_intraday_oi.csv"
    have = set()
    old = pd.DataFrame()
    if path.exists():
        old = pd.read_csv(path, parse_dates=["ts"])
        have = set(old["ts"].dt.date.astype(str))

    today = dt.date.today()
    new = []
    for i in range(back):
        d = today - dt.timedelta(days=i)
        if d.weekday() >= 5:
            continue
        s = d.isoformat()
        # Today is refetched every run: the session is still filling in.
        if s in have and d != today:
            continue
        df = fetch_day(index, s)
        if not df.empty:
            new.append(df)

    if not new and old.empty:
        return pd.DataFrame()
    d = pd.concat(([old] if not old.empty else []) + new, ignore_index=True)
    d = d.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    d.to_csv(path, index=False)
    return d


if __name__ == "__main__":
    for idx in INDEXES:
        d = update(idx)
        if d.empty:
            print(f"{idx:<10} no data")
            continue
        days = d["ts"].dt.date.nunique()
        print(f"{idx:<10} {len(d):>5} points  {days:>3} days  "
              f"{d['ts'].min().date()} -> {d['ts'].max().date()}")
