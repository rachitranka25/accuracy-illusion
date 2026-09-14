#!/usr/bin/env python3
"""Download NIFTY & BANKNIFTY FUTURES (real volume) from Angel.
Daily = ~2.5yr (back-adjusted continuous). 5-min = whatever the front-month
contract has (~3 months). Saves OHLCV CSVs into data_cache/."""
import warnings; warnings.filterwarnings('ignore')
import json, time, urllib.request, datetime as dt
from pathlib import Path
import pandas as pd
from oracle.data import angel as angel_data
from oracle import paths

CACHE = paths.CACHE
MASTER = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

def front_future(master, name):
    futs = [d for d in master if d.get("name") == name and d.get("instrumenttype") == "FUTIDX"
            and d.get("exch_seg") == "NFO"]
    def ed(d):
        try: return dt.datetime.strptime(d["expiry"], "%d%b%Y")
        except Exception: return dt.datetime.max
    futs.sort(key=ed)
    return futs[0] if futs else None            # nearest expiry = front month

def to_df(rows):
    df = pd.DataFrame(rows, columns=["ts","Open","High","Low","Close","Volume"])
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
    return df.set_index("ts")

def fetch(api, token, interval, start, end, step_days):
    out, cur = [], start
    while cur < end:
        chunk_end = min(cur + dt.timedelta(days=step_days), end)
        p = {"exchange":"NFO","symboltoken":str(token),"interval":interval,
             "fromdate":cur.strftime("%Y-%m-%d %H:%M"),
             "todate":chunk_end.strftime("%Y-%m-%d %H:%M")}
        try:
            r = api.getCandleData(p); d = r.get("data") or []
            out += d
        except Exception as e:
            print(f"    chunk {cur.date()} err: {str(e)[:60]}")
        time.sleep(0.5)                          # respect rate limits
        cur = chunk_end
    # dedupe by timestamp
    seen, uniq = set(), []
    for row in out:
        if row[0] not in seen:
            seen.add(row[0]); uniq.append(row)
    return uniq


def main():
    print("loading Angel scrip master…")
    master = json.load(urllib.request.urlopen(MASTER, timeout=40))
    FEED, msg = angel_data.connect()
    print(msg[:55])
    api = FEED.api
    now = dt.datetime.now()

    for name in ["NIFTY", "BANKNIFTY"]:
        f = front_future(master, name)
        if not f:
            print(f"{name}: no future found"); continue
        tok, sym = f["token"], f["symbol"]
        print(f"\n{name} front future: {sym} (token {tok}, expiry {f['expiry']})")

        daily = fetch(api, tok, "ONE_DAY", dt.datetime(2024,1,1), now, 2000)
        dd = to_df(daily)
        dd.to_csv(CACHE / f"{name}_FUT_daily.csv")
        print(f"  daily : {len(dd)} rows  {dd.index[0].date()} → {dd.index[-1].date()}  "
              f"vol>0: {(dd['Volume']>0).mean()*100:.0f}%")

        m5 = fetch(api, tok, "FIVE_MINUTE", now - dt.timedelta(days=95), now, 60)
        d5 = to_df(m5)
        d5.to_csv(CACHE / f"{name}_FUT_5m.csv")
        print(f"  5-min : {len(d5)} rows  {d5.index[0].date()} → {d5.index[-1].date()}  "
              f"vol>0: {(d5['Volume']>0).mean()*100:.0f}%")

    print("\n✅ saved to data_cache/  (NIFTY_FUT_*.csv, BANKNIFTY_FUT_*.csv)")


if __name__ == "__main__":
    main()
