import json
from pathlib import Path

import numpy as np
import pandas as pd
from oracle import paths


HIST = json.loads(Path("forecast_history_v2.json").read_text())
NAMES = ["BANKNIFTY", "NIFTY50"]


def tmin(t):
    return int(t[:2]) * 60 + int(t[3:5])


def prime_ok(t):
    return "09:19" < t[:5] < "14:30" and not ("11:00" <= t[:5] <= "13:30")


def quality(rec, h):
    sig = rec.get("signal", {})
    q = 50 + (sig.get("align", 0) or 0) * 5
    q += (h.get("conf") or 50) - 50
    q += 10 if sig.get("ultra") else 0
    q += 8 if sig.get("hi_conviction") else 0
    q += 7 if sig.get("strong") else 0
    q -= 14 if sig.get("window") == "midday" else 0
    return max(0, min(99, q))


def load_bars(name):
    df = pd.read_csv(paths.CACHE / f"{name}_5m_long.csv")
    df["ts"] = pd.to_datetime(df["ts"], utc=False)
    try:
        df["ts"] = df["ts"].dt.tz_localize(None)
    except Exception:
        df["ts"] = df["ts"].dt.tz_convert(None)
    df["day"] = df["ts"].dt.strftime("%Y-%m-%d")
    df["tm"] = df["ts"].dt.strftime("%H:%M")
    out = {}
    for day, g in df.groupby("day"):
        g = g.sort_values("ts").reset_index(drop=True)
        cl = g["Close"]
        speed = cl.diff().abs().rolling(6).mean() / cl / 5.0
        mom = cl.pct_change(6).abs()
        path = cl.diff().abs().rolling(12).sum()
        eff = (cl - cl.shift(12)).abs() / path
        out[day] = {
            "bars": g[["tm", "Open", "High", "Low", "Close"]].values.tolist(),
            "speed": dict(zip(g["tm"], speed)),
            "mom": dict(zip(g["tm"], mom)),
            "eff": dict(zip(g["tm"], eff.replace([np.inf, -np.inf], np.nan))),
        }
    return out


BARS = {n: load_bars(n) for n in NAMES}


def one_call(rec, tgt_k, sl_k, min_pct, fixed_tstop=None):
    h = rec.get("h", {}).get("30m")
    h60 = rec.get("h", {}).get("60m")
    if not h or not h60:
        return None
    entry = rec.get("price")
    if not entry:
        return None
    ups = rec.get("ups", 2)
    d = "UP" if ups >= 3 else "DOWN" if ups <= 1 else h["dir"]
    base = h60.get("tgt_pts") or h.get("tgt_pts") or 0
    tgt_pts = max(round(base * tgt_k), round(entry * min_pct), 1)
    sl_pts = max(round(base * sl_k), 1)
    if d == "UP":
        tgt, sl = entry + tgt_pts, entry - sl_pts
    else:
        tgt, sl = entry - tgt_pts, entry + sl_pts
    sig = rec.get("signal", {})
    return {
        "time": rec.get("time"), "day": rec.get("day"), "price": entry,
        "dir": d, "tgt": tgt, "sl": sl, "tgt_pts": tgt_pts, "sl_pts": sl_pts,
        "strong": bool(sig.get("strong")), "ultra": bool(sig.get("ultra")),
        "hi_conviction": bool(sig.get("hi_conviction")),
        "align": sig.get("align", 0), "midday": sig.get("window") == "midday",
        "quality": quality(rec, h), "conf": h.get("conf"), "tstop": fixed_tstop,
    }


def outcome(name, call, max_hold):
    daypack = BARS[name].get(call["day"])
    if not daypack:
        return None
    d, tgt, sl, entry = call["dir"], call["tgt"], call["sl"], call["price"]
    t0 = call["time"]
    t0m = tmin(t0)
    seen = False
    last_c = entry
    held = 0
    for b in daypack["bars"]:
        tm, _, hi, lo, c = b
        if tm <= t0:
            continue
        seen = True
        held = tmin(tm) - t0m
        last_c = c
        ht = (d == "UP" and hi >= tgt) or (d == "DOWN" and lo <= tgt)
        hs = (d == "UP" and lo <= sl) or (d == "DOWN" and hi >= sl)
        if ht and hs:
            return "miss", held, -call["sl_pts"]
        if ht:
            return "hit", held, call["tgt_pts"]
        if hs:
            return "miss", held, -call["sl_pts"]
        if held >= max_hold:
            move = (last_c - entry) if d == "UP" else (entry - last_c)
            return ("profit" if move > 0 else "miss"), held, round(move, 1)
    if not seen:
        return None
    move = (last_c - entry) if d == "UP" else (entry - last_c)
    return ("profit" if move > 0 else "miss"), held, round(move, 1)


def run(params):
    tgt_k, sl_k, min_pct, max_hold, qmin, spmin, mommin, effmin, gap, hc_only = params
    allcalls = []
    by_name = {}
    for name in NAMES:
        rows = [r for r in HIST.get(name, []) if "h" in r and r.get("day") in BARS[name]]
        calls = []
        pos_dir = None
        last_day = None
        last_t = None
        for r in rows:
            c = one_call(r, tgt_k, sl_k, min_pct)
            if not c:
                continue
            pack = BARS[name].get(c["day"])
            if not pack:
                continue
            sp = pack["speed"].get(c["time"], np.nan)
            mo = pack["mom"].get(c["time"], np.nan)
            ef = pack["eff"].get(c["time"], np.nan)
            ok = c["strong"] and prime_ok(c["time"]) and c["align"] >= 4 and c["quality"] >= qmin
            if hc_only:
                ok = ok and c["hi_conviction"]
            if ok and np.isfinite(sp):
                ok = sp >= spmin
            if ok and np.isfinite(mo):
                ok = mo >= mommin
            if ok and np.isfinite(ef):
                ok = ef >= effmin
            if not ok:
                continue
            tm = tmin(c["time"])
            if c["day"] != last_day:
                pos_dir, last_t, last_day = None, None, c["day"]
            flip = c["dir"] != pos_dir
            far = last_t is None or tm - last_t >= gap
            if not (flip and far):
                continue
            res = outcome(name, c, max_hold)
            if not res:
                continue
            c["outcome"], c["mins"], c["pts"] = res
            calls.append(c)
            pos_dir, last_t = c["dir"], tm
        by_name[name] = calls
        allcalls += [(name, c) for c in calls]
    resolved = [c for _, c in allcalls]
    green = [c for c in resolved if c["outcome"] in ("hit", "profit")]
    hits = [c for c in resolved if c["outcome"] == "hit"]
    if not resolved:
        return None
    return {
        "params": params,
        "n": len(resolved),
        "green": len(green),
        "green_pct": round(len(green) / len(resolved) * 100, 1),
        "hit_pct": round(len(hits) / len(resolved) * 100, 1),
        "net": round(sum(c["pts"] for c in resolved), 1),
        "avg_min": round(sum(c["mins"] for c in green) / len(green), 1) if green else None,
        "by_name": {n: (len(cs), round(sum(c["outcome"] in ("hit", "profit") for c in cs) / len(cs) * 100, 1) if cs else None,
                         round(sum(c["pts"] for c in cs), 1) if cs else 0)
                    for n, cs in by_name.items()},
    }


grid = []
for tgt_k in [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]:
    for sl_k in [2.4, 3.0, 4.0, 5.0]:
        for min_pct in [0.00015, 0.00025, 0.00035]:
            for max_hold in [10, 15, 20, 30]:
                for qmin in [82, 88, 90, 92]:
                    for spmin in [0.00000, 0.00010, 0.00014, 0.00018]:
                        for mommin in [0.0000, 0.0010, 0.0020, 0.0030]:
                            for effmin in [0.00, 0.12, 0.20]:
                                for gap in [20, 45, 75, 120]:
                                    for hc_only in [False, True]:
                                        r = run((tgt_k, sl_k, min_pct, max_hold, qmin,
                                                 spmin, mommin, effmin, gap, hc_only))
                                        if r and r["n"] >= 8:
                                            grid.append(r)

grid.sort(key=lambda x: (x["green_pct"], x["n"], x["net"]), reverse=True)
for r in grid[:40]:
    print(r)
