"""
Long-history EOD option chain (2016 onward).

The positioning screen ran on 350 days, which is enough to reject a feature but
not enough to trust one. This pulls roughly a decade so the same questions can
be asked with real power -- specifically, whether anything in the option chain
survives once the price component is regressed out. The EOD chain was 0.818
correlated with the 10-day return; the honest follow-up is whether the residual
carries any signal at all.

The site serves this as monthly-expiry chains, one request per trading date, so
it is a few thousand small calls. Kept polite: modest concurrency, on-disk
caching, resumable.
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from oracle import paths

CACHE = paths.CACHE
RAW = CACHE / "oc_raw"
RAW.mkdir(parents=True, exist_ok=True)

SRC = {
    "BANKNIFTY": ("banknifty/bank-nifty-hoc", "bank-nifty-mhoc",
                  "banknifty/bank-nifty-option-chain-historical-data.php"),
    "NIFTY": ("nifty/nifty-hoc", "nifty-mhoc",
              "nifty/nifty-option-chain-historical-data.php"),
}
BASE = "https://tradingtick.in"


def _curl(url, referer):
    out = subprocess.run(
        ["curl", "-s", "-A", "Mozilla/5.0", "-H", "X-Requested-With: XMLHttpRequest",
         "-e", referer, "--max-time", "25", url],
        capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except Exception:
        return None


def _filt(name, **kw):
    d, stem, ref = SRC[name]
    q = "&".join(f"{k}={v}" for k, v in kw.items())
    return _curl(f"{BASE}/{d}/{stem}-filters.php?{q}", f"{BASE}/{ref}") or []


def _data(name, expiry, date):
    d, stem, ref = SRC[name]
    return _curl(f"{BASE}/{d}/{stem}-data.php?expiry={expiry}&date={date}",
                 f"{BASE}/{ref}")


def enumerate_dates(name):
    """(expiry, date) pairs across every year the site exposes."""
    pairs = []
    for year in _filt(name, type="year"):
        for month in _filt(name, type="month", year=year):
            for exp in _filt(name, type="expiryDate", year=year, month=month):
                for date in _filt(name, type="date", expiry=exp):
                    pairs.append((exp, date))
    return sorted(set(pairs), key=lambda x: x[1])


def fetch_one(args):
    name, exp, date = args
    f = RAW / f"{name}_{date}.json"
    if f.exists():
        return None
    j = _data(name, exp, date)
    rows = (j or {}).get("data")
    if not rows:
        return None
    f.write_text(json.dumps(rows))
    return date


def build(name):
    """Collapse per-day chains into one tidy frame."""
    out = []
    for f in sorted(RAW.glob(f"{name}_*.json")):
        date = f.stem.split("_", 1)[1]
        try:
            rows = json.loads(f.read_text())
        except Exception:
            continue
        for r in rows:
            out.append({
                "date": date, "strike": r.get("Strike"), "spot": r.get("Spot"),
                "ce_oi": r.get("OI"), "ce_doi": r.get("OI_Change"),
                "ce_vol": r.get("Volume"), "ce_close": r.get("Close"),
                "pe_oi": r.get("PE_OI"), "pe_doi": r.get("PE_OI_Change"),
                "pe_vol": r.get("PE_Volume"), "pe_close": r.get("PE_Close"),
            })
    d = pd.DataFrame(out)
    if d.empty:
        return d
    d["date"] = pd.to_datetime(d["date"])
    p = CACHE / f"{name}_oc_long.csv"
    d.to_csv(p, index=False)
    return d


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "BANKNIFTY"
    pairs = enumerate_dates(name)
    print(f"{name}: {len(pairs)} trading dates found", flush=True)

    jobs = [(name, e, d) for e, d in pairs]
    done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, r in enumerate(ex.map(fetch_one, jobs), 1):
            if r:
                done += 1
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)} requested, {done} new", flush=True)

    d = build(name)
    if d.empty:
        print("no data assembled")
    else:
        print(f"{name}: {len(d)} strike-rows, {d['date'].nunique()} days "
              f"{d['date'].min().date()} -> {d['date'].max().date()}")
