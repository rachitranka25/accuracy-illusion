#!/usr/bin/env python3
"""Bulk-download NSE F&O bhavcopy (UDiFF) and extract NIFTY + BANKNIFTY index
options & futures EOD (OHLC + Volume + Open Interest). Free, from NSE archives.
Covers ~2 years (UDiFF format began Jul 2024). Saves per-index CSVs."""
import urllib.request, zipfile, io, time, datetime as dt
from pathlib import Path
import pandas as pd
from oracle import paths

CACHE = paths.CACHE
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
      "Accept": "*/*"}
URL = "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d}_F_0000.csv.zip"
KEEP = ["TradDt","FinInstrmTp","TckrSymb","XpryDt","StrkPric","OptnTp",
        "OpnPric","HghPric","LwPric","ClsPric","SttlmPric","UndrlygPric",
        "OpnIntrst","ChngInOpnIntrst","TtlTradgVol","TtlNbOfTxsExctd"]
WANT_SYMS = {"NIFTY", "BANKNIFTY"}
WANT_TP = {"IDO", "IDF"}                       # index options + index futures

def one_day(day):
    url = URL.format(d=day.strftime("%Y%m%d"))
    try:
        raw = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25).read()
    except Exception:
        return None                            # holiday / not published / error
    try:
        z = zipfile.ZipFile(io.BytesIO(raw))
        df = pd.read_csv(z.open(z.namelist()[0]))
        df = df[df["TckrSymb"].isin(WANT_SYMS) & df["FinInstrmTp"].isin(WANT_TP)]
        return df[[c for c in KEEP if c in df.columns]]
    except Exception:
        return None


def main():
    start = dt.date(2024, 7, 8)
    end = dt.date.today()
    day = start
    parts, got, miss = [], 0, 0
    print(f"downloading F&O bhavcopy {start} → {end} …")
    while day <= end:
        if day.weekday() < 5:                      # skip weekends
            r = one_day(day)
            if r is not None and len(r):
                parts.append(r); got += 1
            else:
                miss += 1
            if (got + miss) % 25 == 0:
                print(f"  {day}  files ok:{got} miss:{miss}  rows:{sum(len(p) for p in parts):,}")
            time.sleep(0.4)                        # polite rate limit
        day += dt.timedelta(days=1)

    alldf = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    for sym in WANT_SYMS:
        s = alldf[alldf["TckrSymb"] == sym]
        opt = s[s["FinInstrmTp"] == "IDO"]; fut = s[s["FinInstrmTp"] == "IDF"]
        opt.to_csv(CACHE / f"{sym}_options_eod.csv", index=False)
        fut.to_csv(CACHE / f"{sym}_futures_eod.csv", index=False)
        print(f"{sym}: options {len(opt):,} rows | futures {len(fut):,} rows"
              f" | dates {s['TradDt'].nunique()}")
    print(f"\n✅ done. trading days fetched: {got}, missing/holiday: {miss}")


if __name__ == "__main__":
    main()
