#!/usr/bin/env python3
"""Download NSE daily Participant-wise Open Interest (FII/DII/Pro/Client) — free.
Builds data_cache/participant_oi.csv with each day's index-futures long/short per
participant type. This is 'smart money' positioning, published EOD by NSE."""
import urllib.request, io, time, datetime as dt
from pathlib import Path
import pandas as pd
from oracle import paths

CACHE = paths.CACHE
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
      "Accept": "*/*"}
URL = "https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{d}.csv"


def one_day(day):
    url = URL.format(d=day.strftime("%d%m%Y"))
    try:
        raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                     timeout=25).read().decode("utf-8", "ignore")
    except Exception:
        return None
    try:
        # row 0 is a title; row 1 is the header
        df = pd.read_csv(io.StringIO(raw), skiprows=1)
        df.columns = [c.strip() for c in df.columns]
        df["Client Type"] = df["Client Type"].astype(str).str.strip()
        keep = ["Client Type", "Future Index Long", "Future Index Short",
                "Option Index Call Long", "Option Index Put Long",
                "Option Index Call Short", "Option Index Put Short"]
        df = df[[c for c in keep if c in df.columns]]
        out = {"date": day.strftime("%Y-%m-%d")}
        for _, r in df.iterrows():
            ct = r["Client Type"]
            if ct not in ("Client", "DII", "FII", "Pro"):
                continue
            out[f"{ct}_fut_long"] = r.get("Future Index Long")
            out[f"{ct}_fut_short"] = r.get("Future Index Short")
        return out
    except Exception:
        return None


if __name__ == "__main__":
    start = dt.date(2024, 7, 1)
    end = dt.date.today()
    rows, got, miss, day = [], 0, 0, start
    print(f"downloading participant-OI {start} -> {end} …")
    while day <= end:
        if day.weekday() < 5:
            r = one_day(day)
            if r and any(k.endswith("fut_long") for k in r):
                rows.append(r); got += 1
            else:
                miss += 1
            if (got + miss) % 40 == 0:
                print(f"  {day}  ok:{got} miss:{miss}")
            time.sleep(0.35)
        day += dt.timedelta(days=1)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE / "participant_oi.csv", index=False)
    print(f"\n✅ saved participant_oi.csv — {got} days, {miss} missing")
    if got:
        print(df.tail(3).to_string(index=False))
