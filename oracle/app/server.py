#!/usr/bin/env python3
"""
NIFTY BEAST — paper-trading dashboard with live moving candles.

Serves a local web UI (http://localhost:8181) showing 1-minute candlestick
charts for BankNifty & Nifty50 with the signal engine from nifty_live.py
paper-trading automatically on top: opening range, entry/SL/TGT lines,
trade markers, running paper P&L, and manual BUY/SELL buttons so you can
practice alongside it.

Modes (picked automatically):
  - Market open  → LIVE: real prices, refreshed every ~30 s (free Yahoo data,
    lag measured and displayed).
  - Market closed → shows the last real session statically (charts, trades,
    forecast history). No simulations — real world only.

Run:
  python3 nifty_dashboard.py [--capital 100000] [--risk 1.0] [--port 8181]

Everything is PAPER money until YOU decide otherwise. The point of this
dashboard is to let you verify the real accuracy for free, for as many
sessions as you want, before a single real rupee is at risk.
"""

import argparse
import json
import os
import threading
import time
import webbrowser
from datetime import time as dtime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd

from oracle.features import indicators as nl
from oracle.data import angel as angel_data
from oracle.models import direction as forecast_engine
from oracle.app import ai_analyst
from oracle import paths

HERE = paths.WEB                       # HTML pages live here
LOCK = threading.Lock()
STATE = {}          # index name -> state dict
CONFIG = {"capital": 100000.0, "risk": 1.0}
FEED = None         # AngelFeed when configured, else None -> Yahoo fallback
FC_ENGINES = {}     # per-index per-minute forecast models
# ---- OPT-IN head-to-head experiment (legacy vs new). Off by default; enable
#      with  H2H=1 python3 nifty_dashboard.py  to log both engines for a day. ----
H2H = os.environ.get("H2H") == "1"
LEGACY_ENGINES = {}
H2H_PATH = paths.RUNTIME / "head2head.json"
VOL_ENGINES = {}     # volatility (big-move) LSTM per index — ~68-74% OOS
META_FILTERS = {}    # meta-label model per index (BankNifty only) — see meta_filter.py
VIX_TICKS = []      # ["HH:MM", price] per minute — feature for BANKNIFTY models
ES_TICKS = []       # S&P 500 futures ticks
INR_TICKS = []      # USD/INR ticks
FC_HIST_PATH = paths.RUNTIME / "forecast_history_v2.json"
try:
    FC_HIST = json.loads(FC_HIST_PATH.read_text())
except Exception:
    FC_HIST = {}
MANUAL_SL_PCT = 0.0025          # manual paper trades: SL 0.25% away, TGT 1.5R
CACHE_DIR = paths.CACHE

# ---- AI output log (for later accuracy review) --------------------------
AI_LOG_PATH = paths.RUNTIME / "ai_log.json"
_ai_last = {"analyst": None, "copilot": None}


def log_ai(kind, payload):
    """Append a fresh AI output to ai_log.json with IST time + live prices.
    Only logs when the content actually changed (skips cached repeats)."""
    key = json.dumps(payload, sort_keys=True)
    if _ai_last.get(kind) == key:
        return
    _ai_last[kind] = key
    now = pd.Timestamp.now(tz=nl.TZ)
    with LOCK:
        prices = {n: STATE.get(n, {}).get("price") for n in nl.TICKERS}
    rec = {"ts": now.isoformat(), "time": now.strftime("%Y-%m-%d %I:%M %p"),
           "kind": kind, "prices": prices, "output": payload}
    try:
        log = json.loads(AI_LOG_PATH.read_text()) if AI_LOG_PATH.exists() else []
    except Exception:
        log = []
    log.append(rec)
    AI_LOG_PATH.write_text(json.dumps(log[-2000:], indent=1, ensure_ascii=False))


def _cached(fname, fetchers):
    """Disk-cached market data: fresh cache -> Angel -> Yahoo -> stale cache."""
    f = CACHE_DIR / fname
    today = str(pd.Timestamp.now(tz=nl.TZ).date())
    if f.exists():
        try:
            if pd.Timestamp(f.stat().st_mtime, unit="s", tz="UTC") \
                    .tz_convert(nl.TZ).strftime("%Y-%m-%d") == today:
                df = pd.read_csv(f, index_col=0, parse_dates=True)
                if len(df) > 50:
                    return df
        except Exception:
            pass
    for fetch in fetchers:
        try:
            df = fetch()
            if df is not None and len(df) > 50:
                df.to_csv(f)
                return df
        except Exception:
            continue
    if f.exists():                                 # stale beats nothing
        try:
            return pd.read_csv(f, index_col=0, parse_dates=True)
        except Exception:
            pass
    return None


def fetch_daily_cached(name, symbol):
    return _cached(f"{name}_daily.csv", [
        (lambda: FEED.candles(name, "1d", days=730)) if FEED else (lambda: None),
        lambda: nl.fetch_daily(symbol),
    ])


def fetch_5m_cached(name, symbol):
    return _cached(f"{name}_5m.csv", [
        (lambda: FEED.candles(name, "5m", days=90)) if FEED else (lambda: None),
        lambda: nl.flat(nl.yf.download(symbol, period="60d", interval="5m",
                                       progress=False, auto_adjust=True)),
    ])


def market_open_now():
    now = pd.Timestamp.now(tz=nl.TZ)
    return now.dayofweek < 5 and dtime(9, 15) <= now.time() <= dtime(15, 30)


def new_state(name):
    return {
        "name": name, "mode": "REPLAY", "day": "", "status": "starting…",
        "bias": None, "or": None, "candles": [], "instruction": "loading…",
        "forecast": None, "indicators": [], "pattern": None, "live": None,
        "auto_pos": None, "manual_pos": None, "trades": [],
        "price": None, "lag": None, "finished": False,
    }


def serialize_pos(p, dirkey="dir"):
    if not p:
        return None
    return {"side": "LONG" if p[dirkey] == 1 else "SHORT",
            "entry": p["entry"], "sl": p["sl"], "tgt": p["tgt"], "qty": p["qty"]}


def record(state, kind, side, entry, exit_price, qty, r, pnl, reason, when):
    state["trades"].append({
        "time": when, "kind": kind, "side": side,
        "entry": round(entry, 1), "exit": round(exit_price, 1),
        "qty": round(qty, 2), "r": round(r, 2), "pnl": round(pnl, 0),
        "reason": reason,
    })


def check_manual(state, h, l, c, t):
    mp = state["manual_pos"]
    if not mp:
        return
    d = mp["dir"]
    exit_price = reason = None
    if d == 1 and l <= mp["sl"]:
        exit_price, reason = mp["sl"], "SL"
    elif d == 1 and h >= mp["tgt"]:
        exit_price, reason = mp["tgt"], "TGT"
    elif d == -1 and h >= mp["sl"]:
        exit_price, reason = mp["sl"], "SL"
    elif d == -1 and l <= mp["tgt"]:
        exit_price, reason = mp["tgt"], "TGT"
    elif t >= nl.SQUARE_OFF:
        exit_price, reason = c, "15:15"
    if exit_price is not None:
        pnl = d * (exit_price - mp["entry"]) * mp["qty"]
        r = pnl / (abs(mp["entry"] - mp["sl"]) * mp["qty"])
        record(state, "MANUAL", "LONG" if d == 1 else "SHORT",
               mp["entry"], exit_price, mp["qty"], r, pnl, reason, mp["last_ts"])
        state["manual_pos"] = None


def build_forecast(eng, price, t, just_opened, mfc=None):
    """Forecast card for Beast View using ForecastEngine."""
    if not eng: return {}
    fc = {}
    if mfc and "h" in mfc:
        fc["horizons"] = mfc["h"]
        best_hz = max(mfc["h"].values(), key=lambda x: x["conf"])
        fc["conf"] = round(best_hz["conf"])
        fc["bias"] = "LONG" if best_hz["dir"] == "UP" else "SHORT"
    else:
        fc["conf"] = round(eng.conf * 100)
        fc["bias"] = "LONG" if eng.dir == 1 else "SHORT"
    
    fc["risk_rupees"] = round(eng.risk_rupees)
    side_word = "BUY" if fc["bias"] == "LONG" else "SELL"

    if eng.pos:
        p = eng.pos
        fc.update(
            action=(("BUY NOW!" if p["dir"] == 1 else "SELL NOW!") if just_opened
                    else "HOLD — IN TRADE"),
            trade_now="YES" if just_opened else "IN-TRADE",
            side="LONG" if p["dir"] == 1 else "SHORT",
            qty=round(p["qty"], 1), exposure=round(p["qty"] * price),
            entry=round(p["entry"], 1),
            sl=round(p["sl"], 1), sl_pts=round(abs(p["entry"] - p["sl"])),
            tgt=round(p["tgt"], 1), tgt_pts=round(abs(p["tgt"] - p["entry"])),
            max_loss=round(eng.risk_rupees),
            max_gain=round(nl.TARGET_R * eng.risk_rupees),
            hold="until Target or SL hits — hard exit at 15:15",
            note=("Enter NOW! High confidence forecast received. 🔔" if just_opened else
                  f"Trade is ON (entered at {p['entry']:.0f}). If you took it, "
                  "hold and wait for SL or Target."))
        return fc

    if eng.tradeable:
        fc.update(
            action="WAITING FOR FORECAST",
            trade_now="NO",
            plan_str="Waiting for confidence ≥ 55%",
            side="WAIT",
            qty=0, exposure=0,
            trigger=price,
            sl=price, sl_pts=0,
            tgt=price, tgt_pts=0,
            max_loss=0,
            max_gain=0,
            hold="Hold until SL or Target",
            note="Waiting for the Forecast Engine to emit a high-confidence signal."
        )
    else:
        fc.update(action="NO-TRADE TODAY", side="", trade_now="NO",
                  note="Engine not ready or market closed.")
    return fc

class ForecastTrader:
    def __init__(self, name, capital, risk_pct):
        self.name = name
        self.risk_rupees = capital * risk_pct / 100
        self.pos = None
        self.results = []
        self.result_r = None
        self.dir = 0
        self.conf = 0.0
        self.tradeable = True
        self.or_hi = None
        self.or_lo = None
        self.aligned = 0

    def on_minute(self, ts, o, h, l, c, fc=None):
        self.result_r = None
        
        if self.pos:
            d = self.pos["dir"]
            exit_price = None
            if d == 1:
                if l <= self.pos["sl"]: exit_price = self.pos["sl"]
                elif h >= self.pos["tgt"]: exit_price = self.pos["tgt"]
            else:
                if h >= self.pos["sl"]: exit_price = self.pos["sl"]
                elif l <= self.pos["tgt"]: exit_price = self.pos["tgt"]
                
            if ts.time() >= nl.SQUARE_OFF:
                exit_price = c
                
            if exit_price is not None:
                self.result_r = (d * (exit_price - self.pos["entry"])) / self.pos["risk_pts"]
                self.results.append(self.result_r)
                self.pos = None
                return "EXITED TRADE"

        if self.pos is None and fc and ts.time() < nl.SQUARE_OFF:
            if "h" in fc:
                best_hz = max(fc["h"].values(), key=lambda x: x["conf"])
                self.conf = best_hz["conf"] / 100.0
                self.dir = 1 if best_hz["dir"] == "UP" else -1
                if best_hz["conf"] >= 55:
                    d = self.dir
                    sl_pts = best_hz["sl_pts"]
                    qty = self.risk_rupees / sl_pts if sl_pts > 0 else 0
                    if getattr(self, "user_qty", None):
                        qty = float(self.user_qty)
                    self.pos = {
                        "dir": d, "entry": c, "sl": best_hz["sl"], "tgt": best_hz["tgt"],
                        "qty": qty, "risk_pts": sl_pts, "ts": ts.strftime("%H:%M")
                    }
                    return f"ENTERED {best_hz['dir']} at {c:.1f}"
        return "WAITING"

def _add_min(hhmm, mins):
    h, m = int(hhmm[:2]), int(hhmm[3:5])
    m += mins; h += m // 60; m %= 60
    return f"{h:02d}:{m:02d}"


def save_fc_hist():
    try:
        FC_HIST_PATH.write_text(json.dumps(FC_HIST, indent=2))
    except Exception:
        pass


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _forecast_quality_pick(rec):
    """Best current horizon from one multi-horizon forecast.

    This is deliberately causal: it uses only model tier, calibrated tier accuracy,
    agreement, time window and volatility regime. It never reads the later outcome.
    """
    if not rec or "h" not in rec:
        return None
    sig = rec.get("signal", {}) or {}
    hmap = rec.get("h", {}) or {}
    best = None
    for key, minutes in forecast_engine.HORIZONS:
        if key not in ("5m", "15m"):
            continue
        z = hmap.get(key)
        if not z or z.get("sl") is None:
            continue
        tier = z.get("tier") or "LOW"
        acc = z.get("tier_acc")
        acc_base = float(acc if acc is not None else 50)
        conf = float(z.get("conf") or 50)
        aligned = z.get("dir") == sig.get("dir") or rec.get("consensus") == z.get("dir")
        score = acc_base
        score += (conf - 50) * 0.22
        score += {"HIGH": 7.0, "MEDIUM": 1.5, "LOW": -5.0}.get(tier, 0.0)
        score += 5.0 if aligned else -6.0
        score += float(sig.get("align") or 0) * 1.7
        if sig.get("strong"):
            score += 6.0
        if sig.get("ultra"):
            score += 8.0
        if sig.get("hi_conviction"):
            score += 5.0
        if sig.get("window") == "midday":
            score -= 10.0
        # Prefer actionable intraday horizons, with a tiny nod to speed.
        score += {"5m": 2.0, "15m": 2.5, "30m": 1.5, "60m": 0.0}.get(key, 0.0)
        score = round(_clamp(score, 0, 99), 1)
        tgt_pts = float(z.get("tgt_pts") or 0)
        sl_pts = float(z.get("sl_pts") or 0)
        p = _clamp(acc_base / 100.0, 0.35, 0.70)
        expectancy = round(p * tgt_pts - (1 - p) * sl_pts, 1)
        ppm = round(tgt_pts / max(forecast_engine.HORIZON_MIN.get(key, 5), 1), 2)
        action = "WAIT"
        if (sig.get("ultra") or sig.get("hi_conviction")) and tier == "HIGH" and score >= 72:
            action = "SNIPE"
        elif sig.get("strong") and tier == "HIGH" and score >= 64:
            action = "TAKE"
        grade = "A+" if score >= 82 else "A" if score >= 72 else "B" if score >= 62 else "C"
        row = {
            "time": rec.get("time"), "price": rec.get("price"), "horizon": key,
            "minutes": forecast_engine.HORIZON_MIN.get(key), "dir": z.get("dir"),
            "tier": tier, "tier_acc": acc, "conf": z.get("conf"),
            "score": score, "grade": grade, "action": action,
            "expectancy_pts": expectancy, "ppm": ppm,
            "tgt": z.get("tgt"), "sl": z.get("sl"),
            "tgt_pts": z.get("tgt_pts"), "sl_pts": z.get("sl_pts"),
            "outcome": z.get("outcome"), "move": z.get("move"),
            "strong": bool(sig.get("strong")), "ultra": bool(sig.get("ultra")),
            "hi_conviction": bool(sig.get("hi_conviction")),
            "align": sig.get("align", 0), "window": sig.get("window"),
        }
        if best is None or row["score"] > best["score"]:
            best = row
    return best


def _elite_forecast_stats(rows):
    picks = []
    for r in rows:
        p = _forecast_quality_pick(r)
        if not p or p["action"] == "WAIT" or p["outcome"] not in ("hit", "miss"):
            continue
        picks.append(p)
    green = [p for p in picks if p["outcome"] == "hit"]
    return {
        "n": len(picks),
        "green": len(green),
        "pct": round(len(green) / len(picks) * 100) if picks else None,
        "avg_score": round(sum(p["score"] for p in picks) / len(picks), 1)
                     if picks else None,
        "avg_ppm": round(sum(p["ppm"] for p in picks) / len(picks), 2)
                   if picks else None,
        "rows": picks[-80:],
    }


# ── The Rs 10,000 BOT ──────────────────────────────────────────────────────
# Only the highest-conviction directional setups — where the vol-LSTM predicts a
# BIG move AND price sits on a key level (round number or the opening range). On
# BankNifty these lift the odds of a >=0.4% favourable move (the "70-80% option
# day") from ~18% to ~27%. Each setup is an ATM option BUY, sized from the account
# with real risk limits: never all-in, and the day stops after a set drawdown.
def tmin(t):
    """'HH:MM' -> minutes since midnight (module-level helper)."""
    return int(t[:2]) * 60 + int(t[3:5])


_BOT_STEP = {"BANKNIFTY": 100, "NIFTY50": 50, "FINNIFTY": 50}
_BOT_LOT = {"BANKNIFTY": 15, "NIFTY50": 75, "FINNIFTY": 65}
_BOT_DEPLOY = 0.35        # max fraction of equity risked into one setup
_BOT_DAYSTOP = 0.60       # stop trading for the day below 60% of day-start equity


def _bot_levels(candles, step):
    """Key levels a customer can see: round numbers near price + opening-range H/L."""
    if not candles:
        return []
    last = candles[-1][4]
    lv = [round(last / step) * step, (round(last / step) + 1) * step,
          (round(last / step) - 1) * step]
    opn = [c for c in candles if c[0] <= "09:30"]      # first 15 min range
    if len(opn) >= 2:
        lv += [max(c[2] for c in opn), min(c[3] for c in opn)]
    return lv


def _bot_trades(name, records, candles, veng, day, capital=10000):
    """Return the bot's BIG-MOVE option trades for today + an equity curve."""
    step = _BOT_STEP.get(name, 50)
    lot = _BOT_LOT.get(name, 75)
    equity = float(capital)
    day_start = equity
    trades = []
    pos_dir = None
    last_t = None
    vcache = {}

    def vol_at(t):
        key = t[:4]
        if key in vcache:
            return vcache[key]
        upto = [b for b in candles if b[0] <= t]
        pv = veng.predict(upto, day) if (veng and len(upto) > 5) else None
        vcache[key] = pv
        return pv

    def prime(t):
        hm = t[:5]
        return "09:19" < hm < "14:15"

    for r in records:
        if "h" not in r:
            continue
        t = r.get("time")
        entry = r.get("price")
        if not t or not entry or not prime(t):
            continue
        sig = r.get("signal", {})
        if not sig.get("strong"):
            continue
        ups = r.get("ups", 2)
        d = "UP" if ups >= 3 else "DOWN" if ups <= 1 else sig.get("dir")
        if d not in ("UP", "DOWN"):
            continue
        # GATE 1 — vol model must predict a BIG move
        pv = vol_at(t)
        if not pv or pv.get("vol") != "BIG":
            continue
        # GATE 2 — price must sit on a key level
        levels = _bot_levels([c for c in candles if c[0] <= t], step)
        near = next((lv for lv in levels
                     if lv and abs(entry - lv) / entry < 0.0015), None)
        if near is None:
            continue
        tm = tmin(t)
        if d == pos_dir and last_t is not None and tm - last_t < 20:
            continue
        # risk-managed sizing: buy ATM option, deploy <= 35% of equity
        atm = round(entry / step) * step
        prem_pts = max(entry * 0.0035, 1)          # ~ATM weekly premium
        lot_cost = prem_pts * lot
        if equity <= day_start * _BOT_DAYSTOP:      # daily stop hit -> done
            break
        lots = int((equity * _BOT_DEPLOY) // lot_cost)
        if lots < 1:
            continue
        deploy = lots * lot_cost
        # outcome: favourable / adverse index move -> option % return (leveraged)
        tgt_pts = entry * (pv.get("prob", 60) / 100.0) * 0.006 + entry * 0.003
        sl_pts = entry * 0.003
        opt = _bot_option_outcome(candles, t, d, entry, tgt_pts, sl_pts, prem_pts)
        pnl = round(deploy * opt["ret"])
        if opt["outcome"] != "open":
            equity += pnl
        trades.append({
            "time": t, "level": round(near), "level_lbl": _bot_level_label(near, step, entry),
            "dir": d, "option": f"{atm} {'CE' if d == 'UP' else 'PE'}",
            "entry": round(entry, 1), "lots": lots,
            "premium": round(prem_pts), "deploy": round(deploy),
            "conf": pv.get("prob"), "outcome": opt["outcome"],
            "ret_pct": round(opt["ret"] * 100), "pnl": pnl,
            "mins": opt["mins"], "equity": round(equity),
        })
        pos_dir, last_t = d, tm

    resolved = [t for t in trades if t["outcome"] != "open"]
    wins = [t for t in resolved if t["pnl"] > 0]
    return {
        "capital": capital, "equity": round(equity),
        "day_return_pct": round((equity / day_start - 1) * 100, 1),
        "n": len(resolved), "wins": len(wins),
        "win_pct": round(len(wins) / len(resolved) * 100) if resolved else None,
        "open": sum(1 for t in trades if t["outcome"] == "open"),
        "trades": trades,
        "note": "BIG-move setups only · ATM option buy · max 35% deployed/trade · day stops at −40%",
    }


def _bot_level_label(lv, step, price):
    if abs(lv - round(price / step) * step) < step * 0.5 and lv % step == 0:
        return f"round {int(lv)}"
    return f"OR {'high' if lv > price else 'low'}"


def _bot_option_outcome(candles, t0, d, entry, tgt_pts, sl_pts, prem_pts):
    """Simulate the option buy: index path -> leveraged option return."""
    tgt = entry + tgt_pts if d == "UP" else entry - tgt_pts
    sl = entry - sl_pts if d == "UP" else entry + sl_pts
    t0m = tmin(t0)
    seen = False
    for b in candles:
        if b[0] <= t0:
            continue
        seen = True
        held = tmin(b[0]) - t0m
        hi, lo = b[2], b[3]
        ht = (d == "UP" and hi >= tgt) or (d == "DOWN" and lo <= tgt)
        hs = (d == "UP" and lo <= sl) or (d == "DOWN" and hi >= sl)
        if hs and ht:
            return {"outcome": "loss", "ret": -0.40, "mins": held}
        if ht:                                   # favourable move -> big option gain
            ret = min((tgt_pts * 0.6) / prem_pts, 2.0)     # delta~0.6, capped +200%
            return {"outcome": "win", "ret": ret, "mins": held}
        if hs:                                   # adverse -> stop, cushioned by convexity
            return {"outcome": "loss", "ret": max(-(sl_pts * 0.5) / prem_pts, -0.85),
                    "mins": held}
        if held >= 45:                            # theta time-stop
            move = (b[4] - entry) if d == "UP" else (entry - b[4])
            ret = max(min((move * 0.55) / prem_pts, 2.0) - 0.10, -0.85)  # -10% theta drag
            return {"outcome": "win" if ret > 0 else "loss", "ret": ret, "mins": held}
    if not seen:
        return {"outcome": "open", "ret": 0.0, "mins": None}
    return {"outcome": "open", "ret": 0.0, "mins": None}


def update_forecast_stream(state):
    """Emit one forecast per minute per index; score old ones against reality."""
    name = state["name"]
    fce = FC_ENGINES.get(name)
    if not fce:
        return
    candles = state["candles"]
    day = state["day"] or str(pd.Timestamp.now(tz=nl.TZ).date())
    if state.get("mode") != "LIVE" and not state.get("mode") == "CLOSED":
        return                       # skip if neither live nor closed/replay
    hist = FC_HIST.setdefault(name, [])
    changed = False
    fc = None
    other_st = STATE.get(forecast_engine.OTHER.get(name, ""), {})
    other = other_st.get("all_candles") or other_st.get("candles")
    try:
        fc = fce.forecast(candles, day, other_candles=other,
                          vix_ticks=list(VIX_TICKS),
                          es_ticks=list(ES_TICKS),
                          inr_ticks=list(INR_TICKS))
    except Exception as e:
        print(f"FC ERROR for {name}: {e}")
        pass
    if fc and not any(r["time"] == fc["time"] and r.get("day") == day
                      for r in hist[-450:]):
        fc["day"] = day
        fc["mode"] = state["mode"]
        hist.append(fc)
        del hist[:-1500]
        changed = True
    state["mforecast"] = fc or state.get("mforecast")
    closes = {b[0]: b[4] for b in candles}
    times = [b[0] for b in candles]
    for row in hist[-200:]:
        if row.get("day") != day or "h" not in row:
            continue
        for key, hz in row["h"].items():
            if hz["outcome"] is not None:
                continue
            target = _add_min(row["time"], forecast_engine.HORIZON_MIN[key])
            hit = next((t for t in times if t >= target), None)
            if hit:
                d = 1 if hz["dir"] == "UP" else -1
                move = closes[hit] - row["price"]
                hz["move"] = round(move, 1)
                hz["outcome"] = "hit" if d * move > 0 else "miss"
                changed = True
    if changed:
        save_fc_hist()


def feed_bar(state, eng, ts, o, h, l, c):
    """Push one 1-minute bar through engine + manual book, update state."""
    when = ts.strftime("%H:%M")
    # DATA HYGIENE: reject a bad tick (a flat O=H=L=C bar that jumps >0.5% from the
    # previous close). One of these faked a "+507 pt HIT" across 26 forecast rows.
    prev = state["candles"][-1][4] if state["candles"] else None
    if prev and abs(c - prev) > prev * 0.005 and (h - l) < prev * 1e-5:
        state["price"] = round(prev, 1)
        return
    state["candles"].append([when, round(o, 1), round(h, 1), round(l, 1), round(c, 1)])
    state["candles"] = state["candles"][-400:]
    state["indicators"], state["pattern"] = indicator_snapshot(state["candles"])
    want = "bull" if eng.dir == 1 else "bear"
    # adaptive unlock is defined as 4/4 of the ORIGINAL 4 indicators only —
    # the extra market-state chips (Bollinger/Stochastic/Volatility) are UI-only
    # and must NOT loosen the "4/4" gate. (Bug caught 2026-07: card grew 4→7.)
    _CORE_IND = {"RSI(14)", "EMA 9/21", "MACD", "VWAP"}
    eng.aligned = sum(1 for i in state["indicators"]
                      if i["name"] in _CORE_IND and i["verdict"] == want)
    update_forecast_stream(state)
    prev_pos = eng.pos
    mfc = state.get("mforecast")
    eng.user_qty = state.get("user_qty")
    line = eng.on_minute(ts, o, h, l, c, mfc)
    if prev_pos is None and eng.pos:
        pass                                        # opened — shown via auto_pos
    if prev_pos and eng.pos is None and eng.result_r is not None:
        r = eng.result_r
        d = prev_pos["dir"]
        exit_price = prev_pos["entry"] + d * r * prev_pos["risk_pts"]
        reason = ("TGT" if r > 1 else "SL" if r <= -0.9 else "15:15")
        record(state, "AUTO", "LONG" if d == 1 else "SHORT", prev_pos["entry"],
               exit_price, prev_pos["qty"], r, r * eng.risk_rupees, reason, when)
    if state["manual_pos"]:
        state["manual_pos"]["last_ts"] = when
    check_manual(state, h, l, c, ts.time())
    state["forecast"] = build_forecast(eng, c, ts.time(),
                                       prev_pos is None and eng.pos is not None,
                                       mfc)
    state["price"] = round(c, 1)
    state["instruction"] = line
    state["or"] = ({"hi": round(eng.or_hi, 1), "lo": round(eng.or_lo, 1)}
                   if eng.or_hi is not None else None)
    state["auto_pos"] = None   # auto paper-trader disabled (Pulse is the trade layer)
    state["bias"] = {"side": "LONG" if eng.dir == 1 else "SHORT",
                     "conf": round(eng.conf * 100),
                     "tradeable": eng.tradeable}


def on_stream_tick(name, px):
    """Called by the Angel websocket on every tick — instant chart update."""
    if not market_open_now():
        return                       # no ticks, candles or forecasts after 15:30
    minute = pd.Timestamp.now(tz=nl.TZ).strftime("%H:%M")
    if name == "INDIAVIX":
        with LOCK:
            if VIX_TICKS and VIX_TICKS[-1][0] == minute:
                VIX_TICKS[-1][1] = px
            else:
                VIX_TICKS.append([minute, px])
                del VIX_TICKS[:-450]
        return
    with LOCK:
        st = STATE.get(name)
        if not st or st["mode"] != "LIVE":
            return
        if not (st["candles"] and minute <= st["candles"][-1][0]):
            lv = st.get("live")
            if lv and lv[0] == minute:
                lv[2] = round(max(lv[2], px), 1)
                lv[3] = round(min(lv[3], px), 1)
                lv[4] = round(px, 1)
            else:
                # minute rolled over: the old live bar is complete — forecast NOW
                if lv and (not st["candles"] or lv[0] > st["candles"][-1][0]):
                    fce = FC_ENGINES.get(name)
                    if fce:
                        try:
                            day = st["day"] or str(pd.Timestamp.now(tz=nl.TZ).date())
                            other_st = STATE.get(forecast_engine.OTHER.get(name, ""), {})
                            fc = fce.forecast(st["candles"] + [lv], day,
                                              other_candles=other_st.get("all_candles") or other_st.get("candles"),
                                              vix_ticks=list(VIX_TICKS),
                                              es_ticks=list(ES_TICKS),
                                              inr_ticks=list(INR_TICKS))
                            hist = FC_HIST.setdefault(name, [])
                            if fc and not any(
                                    r["time"] == fc["time"] and r.get("day") == day
                                    for r in hist[-450:]):
                                fc["day"] = day
                                fc["mode"] = st["mode"]
                                hist.append(fc)
                                del hist[:-1500]
                                st["mforecast"] = fc
                                save_fc_hist()
                        except Exception:
                            pass
                st["live"] = [minute, round(px, 1), round(px, 1),
                              round(px, 1), round(px, 1)]
        st["price"] = round(px, 1)


def make_engine(name, day, feats):
    eng = ForecastTrader(name, CONFIG["capital"], CONFIG["risk"])
    return eng


def indicator_snapshot(candles):
    """Live indicator chips + candlestick pattern from the 1-minute candles."""
    if len(candles) < 30:
        return [], None
    df = pd.DataFrame(candles, columns=["t", "o", "h", "l", "c"])
    c = df["c"]
    r = float(nl.rsi(c).iloc[-1])
    e9 = float(c.ewm(span=9).mean().iloc[-1])
    e21 = float(c.ewm(span=21).mean().iloc[-1])
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    hist = float((macd - macd.ewm(span=9).mean()).iloc[-1])
    tp = (df["h"] + df["l"] + df["c"]) / 3
    vwap = float(tp.expanding().mean().iloc[-1])   # session VWAP proxy (indices have no volume)
    px = float(c.iloc[-1])
    # Bollinger position (same feature the model uses: bb_pos)
    ma20 = float(c.rolling(20).mean().iloc[-1])
    sd20 = float(c.rolling(20).std().iloc[-1])
    bbpos = (px - ma20) / (2 * sd20) if sd20 > 0 else 0.0
    # Stochastic %K
    lo14 = float(df["l"].rolling(14).min().iloc[-1])
    hi14 = float(df["h"].rolling(14).max().iloc[-1])
    stoch = (px - lo14) / (hi14 - lo14) * 100 if hi14 > lo14 else 50.0
    # ATR volatility state (model uses atr_pct)
    tr = pd.concat([df["h"] - df["l"], (df["h"] - c.shift()).abs(),
                    (df["l"] - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    atr_now, atr_med = float(atr.iloc[-1]), float(atr.median())
    vol_state = ("high" if atr_now > atr_med * 1.3 else
                 "low" if atr_now < atr_med * 0.7 else "normal")
    ind = [
        {"name": "RSI(14)", "value": f"{r:.0f}",
         "verdict": "bull" if r > 55 else "bear" if r < 45 else "neutral"},
        {"name": "EMA 9/21", "value": "9 above" if e9 > e21 else "9 below",
         "verdict": "bull" if e9 > e21 else "bear"},
        {"name": "MACD", "value": "rising" if hist > 0 else "falling",
         "verdict": "bull" if hist > 0 else "bear"},
        {"name": "VWAP", "value": "above" if px > vwap else "below",
         "verdict": "bull" if px > vwap else "bear"},
        {"name": "Bollinger", "value": "upper half" if bbpos > 0.05 else
         "lower half" if bbpos < -0.05 else "mid-band",
         "verdict": "bull" if bbpos > 0.05 else "bear" if bbpos < -0.05 else "neutral"},
        {"name": "Stochastic", "value": f"{stoch:.0f}",
         "verdict": "bull" if stoch > 55 else "bear" if stoch < 45 else "neutral"},
        {"name": "Volatility (ATR)", "value": vol_state, "verdict": "neutral"},
    ]
    o1, h1, l1, c1 = (float(df.iloc[-1][k]) for k in ("o", "h", "l", "c"))
    o2, c2 = float(df.iloc[-2]["o"]), float(df.iloc[-2]["c"])
    body1, body2, rng1 = c1 - o1, c2 - o2, h1 - l1
    pattern = None
    if rng1 > 0 and abs(body1) / rng1 < 0.1:
        pattern = "Doji (indecision)"
    if (min(o1, c1) - l1) > 2 * abs(body1) and (h1 - max(o1, c1)) < abs(body1):
        pattern = "Hammer (bullish)"
    if (h1 - max(o1, c1)) > 2 * abs(body1) and (min(o1, c1) - l1) < abs(body1):
        pattern = "Shooting Star (bearish)"
    if body1 > 0 and body2 < 0 and c1 > o2 and o1 < c2:
        pattern = "Bullish Engulfing"
    if body1 < 0 and body2 > 0 and c1 < o2 and o1 > c2:
        pattern = "Bearish Engulfing"
    return ind, pattern


def prior_levels(daily, session_day):
    """Prev-session H/L/C + pivot for the AI's technical read."""
    try:
        prev = daily[daily.index.date < session_day].iloc[-1]
        h, l, c = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
        p = (h + l + c) / 3
        return {"prev_high": round(h, 1), "prev_low": round(l, 1),
                "prev_close": round(c, 1), "pivot": round(p, 1),
                "r1": round(2 * p - l, 1), "s1": round(2 * p - h, 1)}
    except Exception:
        return None


def init_fc_engine(name, symbol):
    if name not in FC_ENGINES:
        try:
            m5 = fetch_5m_cached(name, symbol)
            FC_ENGINES[name] = forecast_engine.ForecastEngine(
                name, CONFIG["capital"], CONFIG["risk"], m5=m5)
        except Exception as exc:
            print(f"{name}: forecast models unavailable ({exc})")
    if name not in VOL_ENGINES:                # volatility (big-move) LSTM
        try:
            from oracle.models import volatility
            VOL_ENGINES[name] = volatility_lstm.VolatilityLSTM(name)
        except Exception as exc:
            print(f"{name}: volatility engine unavailable ({exc})")
    if name not in META_FILTERS:               # meta-label filter (BankNifty)
        try:
            from oracle.models import meta_filter
            mf = meta_filter.MetaFilter(name)
            META_FILTERS[name] = mf
            if mf.ok:
                print(f"  {name}: meta-filter loaded (keeps top "
                      f"{int(meta_filter.KEEP_TOP * 100)}%, ~{mf.acc:+.2f} pts/min OOS)")
        except Exception as exc:
            print(f"{name}: meta-filter unavailable ({exc})")
    if H2H and name not in LEGACY_ENGINES:     # opt-in: old "Beast" for head-to-head
        try:
            import sys
            exp = str(paths.RESEARCH / "archive")
            if exp not in sys.path:
                sys.path.insert(0, exp)
            import legacy_engine
            LEGACY_ENGINES[name] = legacy_engine.LegacyForecastEngine(
                name, CONFIG["capital"], CONFIG["risk"])
        except Exception as exc:
            print(f"{name}: legacy engine unavailable ({exc})")


def _append_log(path, rec):
    """Quick JSON append (brief lock; only file I/O, no model work)."""
    try:
        with LOCK:
            log = json.loads(path.read_text()) if path.exists() else []
            log.append(rec)
            path.write_text(json.dumps(log[-4000:], ensure_ascii=False))
    except Exception:
        pass


def h2h_logger(name):
    """OPT-IN head-to-head logger (H2H=1). Own thread, ~3 min. Holds LOCK only to
    COPY data; the heavy legacy inference runs OUTSIDE the lock so the live feed is
    never blocked. Logs new-vs-legacy per-horizon directions to head2head.json."""
    time.sleep(90)
    other_name = forecast_engine.OTHER.get(name, "")
    while True:
        try:
            with LOCK:
                st = STATE.get(name, {})
                candles = list(st.get("candles", []))
                day = st.get("day"); price = st.get("price"); mode = st.get("mode")
                new_fc = st.get("mforecast")
                ost = STATE.get(other_name, {})
                other = list(ost.get("all_candles") or ost.get("candles") or [])
                vix = list(VIX_TICKS)
            if (mode == "LIVE" and day and len(candles) >= 15
                    and new_fc and new_fc.get("h")):
                now = pd.Timestamp.now(tz=nl.TZ)
                leg = LEGACY_ENGINES.get(name)
                if leg:
                    lf = leg.forecast(candles, day, other_candles=other, vix_ticks=vix)
                    if lf:
                        _append_log(H2H_PATH, {
                            "ts": now.isoformat(), "time": now.strftime("%I:%M %p"),
                            "index": name, "day": day, "price": price,
                            "new": {k: new_fc["h"][k]["dir"] for k in new_fc["h"]},
                            "legacy": {k: lf["h"][k]["dir"] for k in lf["h"]}})
        except Exception:
            pass
        time.sleep(180)


def closed_worker(name, symbol):
    """Market closed: show the LAST real session statically — no replay."""
    try:
        _closed_worker(name, symbol)
    except Exception as exc:
        with LOCK:
            STATE[name]["status"] = f"load failed: {str(exc)[:90]}"


def _closed_worker(name, symbol):
    init_fc_engine(name, symbol)
    daily = fetch_daily_cached(name, symbol)
    feats = nl.build_daily_features(daily)
    source = "Yahoo (delayed)"
    m1 = None
    if FEED:
        try:
            m1 = FEED.candles(name, "1m", days=4)
            source = "Angel One"
        except Exception:
            m1 = None
    if m1 is None or m1.empty:
        m1 = nl.flat(nl.yf.download(symbol, period="5d", interval="1m",
                                    progress=False, auto_adjust=True))
    day = sorted(set(m1.index.date))[-1]
    bars = m1[m1.index.date == day]
    
    global VIX_TICKS, ES_TICKS, INR_TICKS
    with LOCK:
        if not VIX_TICKS:
            vix = nl.flat(nl.yf.download("^INDIAVIX", period="5d", interval="1m", progress=False, auto_adjust=True))
            vix_bars = vix[vix.index.date == day]
            VIX_TICKS = [[ts.strftime("%H:%M"), float(r["Close"])] for ts, r in vix_bars.iterrows()]
            
            es = nl.flat(nl.yf.download("ES=F", period="5d", interval="1m", progress=False, auto_adjust=True))
            if not es.empty and not es[es.index.date == day].empty:
                es_bars = es[es.index.date == day]
                ES_TICKS = [[ts.strftime("%H:%M"), float(r["Close"])] for ts, r in es_bars.iterrows()]
                
            inr = nl.flat(nl.yf.download("INR=X", period="5d", interval="1m", progress=False, auto_adjust=True))
            if not inr.empty and not inr[inr.index.date == day].empty:
                inr_bars = inr[inr.index.date == day]
                INR_TICKS = [[ts.strftime("%H:%M"), float(r["Close"])] for ts, r in inr_bars.iterrows()]

    eng = make_engine(name, day, feats)
    with LOCK:
        st = STATE[name] = new_state(name)
        st.update(mode="CLOSED", day=str(day), lag=0, finished=True,
                  status=f"market closed — showing last session {day} "
                         f"({source}) · live again Mon-Fri 9:15 AM")
        st["levels"] = prior_levels(daily, day)
        st["all_candles"] = [[ts.strftime("%H:%M"), float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])] for ts, r in bars.iterrows()]
    
    # Wait until both threads have populated all_candles
    while not STATE.get(forecast_engine.OTHER.get(name, ""), {}).get("all_candles"):
        time.sleep(0.1)

    for ts, bar in bars.iterrows():
        feed_bar(st, eng, ts, float(bar["Open"]), float(bar["High"]),
                 float(bar["Low"]), float(bar["Close"]))


def live_worker(name, symbol):
    time.sleep(list(nl.TICKERS).index(name) * 8)   # stagger API calls per index
    init_fc_engine(name, symbol)
    daily = fetch_daily_cached(name, symbol)
    feats = nl.build_daily_features(daily)
    now = pd.Timestamp.now(tz=nl.TZ)
    eng = make_engine(name, now.date(), feats)
    with LOCK:
        st = STATE[name] = new_state(name)
        st.update(mode="LIVE", day=str(now.date()), status="live — connecting…")
        st["levels"] = prior_levels(daily, now.date())
    seen = None
    angel_fails = 0
    last_bars = 0.0
    while True:
        now = pd.Timestamp.now(tz=nl.TZ)
        if now.time() > dtime(15, 30):
            with LOCK:
                STATE[name]["finished"] = True
                STATE[name]["status"] = "market closed — session over"
            return
        # ---- completed 1-minute candles (feeds the engine) every ~30 s ----
        if seen is None or time.time() - last_bars >= 30:
            try:
                if FEED and angel_fails < 3:
                    m1 = FEED.candles(name, "1m", days=1)
                    angel_fails = 0
                    src = "broker real-time data"
                else:
                    m1 = nl.flat(nl.yf.download(symbol, period="1d", interval="1m",
                                                progress=False, auto_adjust=True))
                    src = "Yahoo fallback (delayed) — retrying Angel soon"
                    if FEED:
                        angel_fails -= 1          # decay so Angel gets retried
                m1 = m1[m1.index.date == now.date()]
                last_bars = time.time()
                if not m1.empty:
                    lag = (now - m1.index[-1]).total_seconds() / 60
                    new = m1 if seen is None else m1[m1.index > seen]
                    new = new[new.index < now.floor("min")]   # completed bars only
                    with LOCK:
                        st = STATE[name]
                        st["lag"] = round(lag, 1)
                        if lag > nl.MAX_LAG_MIN:
                            st["status"] = f"⚠️ data {lag:.0f} min stale — not signalling"
                        else:
                            st["status"] = f"live · {src}"
                            for ts, bar in new.iterrows():
                                feed_bar(st, eng, ts, float(bar["Open"]),
                                         float(bar["High"]), float(bar["Low"]),
                                         float(bar["Close"]))
                        lv = st.get("live")       # drop forming candle once covered
                        if lv and st["candles"] and lv[0] <= st["candles"][-1][0]:
                            st["live"] = None
                    if len(new):
                        seen = new.index[-1]
            except Exception as exc:
                angel_fails += 1
                with LOCK:
                    STATE[name]["status"] = \
                        f"candle fetch failed; retrying ({str(exc)[:70]})"
        # ---- live tick every ~3 s (moves the forming candle) ----
        if FEED:
            try:
                px = FEED.ltp(name)
                minute = now.strftime("%H:%M")
                with LOCK:
                    st = STATE[name]
                    if not (st["candles"] and minute <= st["candles"][-1][0]):
                        lv = st.get("live")
                        if lv and lv[0] == minute:
                            lv[2] = round(max(lv[2], px), 1)
                            lv[3] = round(min(lv[3], px), 1)
                            lv[4] = round(px, 1)
                        else:
                            st["live"] = [minute, round(px, 1), round(px, 1),
                                          round(px, 1), round(px, 1)]
                    st["price"] = round(px, 1)
            except Exception:
                pass                              # tick misses are harmless
        time.sleep(1)                             # streamed ticks make this cheap


def ai_snapshot():
    """Full technical state as text — the input for both AI endpoints."""
    with LOCK:
        parts = []
        for name, st in STATE.items():
            b = st.get("bias") or {}
            f = st.get("forecast") or {}
            inds = ", ".join(f"{i['name']}={i['value']}({i['verdict']})"
                             for i in st.get("indicators", []))
            cs = st.get("candles", [])
            day_stats = ""
            if len(cs) > 3:
                highs = max(c[2] for c in cs)
                lows = min(c[3] for c in cs)
                opn = cs[0][1]
                last = cs[-1][4]
                m30 = cs[-30][4] if len(cs) >= 30 else opn
                day_stats = (f"day open={opn} high={highs} low={lows} "
                             f"now={last} last-30min-change={last - m30:+.0f}pts")
            orr = st.get("or") or {}
            lv = st.get("levels") or {}
            mf = st.get("mforecast") or {}
            fcst = ""
            if mf.get("h"):
                fcst = " ".join(f"{k}:{v['dir']}{v['conf']}%"
                                for k, v in mf["h"].items())
                fcst += f" consensus={mf.get('consensus')}"
            hist = FC_HIST.get(name, [])
            today = [r for r in hist
                     if r.get("day") == st.get("day") and "h" in r]
            accs = []
            for key, _k in forecast_engine.HORIZONS:
                res = [r["h"][key] for r in today
                       if r["h"][key]["outcome"] is not None]
                hits = sum(1 for z in res if z["outcome"] == "hit")
                if res:
                    accs.append(f"{key}:{hits}/{len(res)}")
            trades = "; ".join(
                f"{t['time']} {t['side']} {t['r']:+.2f}R ({t['reason']})"
                for t in st.get("trades", [])) or "none"
            parts.append(
                f"== {name} ==\n"
                f"{day_stats}\n"
                f"opening-range: hi={orr.get('hi')} lo={orr.get('lo')}\n"
                f"prior session: {lv}\n"
                f"daily-model bias: {b.get('side')} conf={b.get('conf')}% "
                f"| engine says: {f.get('action')}\n"
                f"indicators: {inds} | pattern: {st.get('pattern')}\n"
                f"multi-horizon forecast now: {fcst}\n"
                f"forecast hit-rates today: {' '.join(accs) or 'collecting'}\n"
                f"paper trades today: {trades}")
    return "\n".join(parts)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                       # silence request logging
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/callback":
            # broker OAuth redirect lands here; show whatever token/code came back
            params = {k: v[0] for k, v in q.items()}
            rows = "".join(f"<tr><td style='padding:6px 12px;color:#898781'>{k}</td>"
                           f"<td style='padding:6px 12px;font-family:monospace'>{v}</td></tr>"
                           for k, v in params.items()) or \
                   "<tr><td style='padding:6px 12px'>no parameters received</td></tr>"
            body = (f"<!DOCTYPE html><html><body style='background:#0d0d0d;color:#fff;"
                    f"font-family:system-ui;padding:40px'>"
                    f"<h2>✅ Broker login redirect received</h2>"
                    f"<p style='color:#c3c2b7'>Copy the code/token below — this is what "
                    f"the API needs to generate your access token:</p>"
                    f"<table style='background:#1a1a19;border-radius:10px'>{rows}</table>"
                    f"<p style='color:#898781'>Keep this private. Never share it with "
                    f"anyone or paste it on any website.</p></body></html>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path in ("/", "/quantum", "/forecasts", "/single", "/pulse", "/volatility"):
            page = ("quantum_live.html" if url.path == "/quantum"
                    else "forecast_feed.html" if url.path == "/forecasts"
                    else "single_feed.html" if url.path == "/single"
                    else "pulse_view.html" if url.path == "/pulse"
                    else "volatility_view.html" if url.path == "/volatility"
                    else "dashboard.html")
            body = (HERE / page).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/forecasts":
            with LOCK:
                out = {}
                for name in nl.TICKERS:
                    hist = FC_HIST.get(name, [])
                    day = STATE.get(name, {}).get("day")
                    today = [r for r in hist
                             if r.get("day") == day and "h" in r]
                    acc = {}
                    for key, _k in forecast_engine.HORIZONS:
                        res = [r["h"][key] for r in today
                               if r["h"][key]["outcome"] is not None]
                        hits = sum(1 for z in res if z["outcome"] == "hit")
                        acc[key] = {"n": len(res), "hits": hits,
                                    "pct": round(hits / len(res) * 100)
                                           if res else None}
                    fce = FC_ENGINES.get(name)
                    current = STATE.get(name, {}).get("mforecast")
                    elite_stats = _elite_forecast_stats(today)
                    out[name] = {
                        "current": current,
                        "best_now": _forecast_quality_pick(current),
                        "elite": elite_stats,
                        "history": today[-400:],
                        "acc": acc,
                        "backtest_acc": fce.backtest_acc if fce else {},
                        "price": STATE.get(name, {}).get("price"),
                        "mode": STATE.get(name, {}).get("mode"),
                    }
            self._json(out)
        elif url.path == "/api/bot":
            # The Rs 10,000 bot — only BIG-move-at-a-level option-buy setups,
            # risk-sized from the account. See _bot_trades.
            cap = float(parse_qs(url.query).get("cap", ["10000"])[0] or 10000)
            with LOCK:
                out = {}
                for name in nl.TICKERS:
                    st = STATE.get(name, {})
                    day = st.get("day")
                    candles = list(st.get("candles", []))
                    records = [r for r in FC_HIST.get(name, [])
                               if r.get("day") == day and "h" in r]
                    veng = VOL_ENGINES.get(name)
                    try:
                        out[name] = _bot_trades(name, records, candles, veng, day, cap)
                    except Exception as exc:
                        out[name] = {"error": str(exc)[:120], "trades": [],
                                     "capital": cap, "equity": cap}
                    out[name]["price"] = st.get("price")
                    out[name]["mode"] = st.get("mode")
            self._json(out)
        elif url.path == "/api/single":
            # ONE clean forecast per minute: consensus direction + 30m realistic
            # bracket (good margin). Honest running hit-rate. Scored on the 30m
            # outcome already stored in history.
            with LOCK:
                out = {}
                for name in nl.TICKERS:
                    st = STATE.get(name, {})
                    hist = FC_HIST.get(name, [])
                    day = st.get("day")
                    today = [r for r in hist if r.get("day") == day and "h" in r]

                    def single_of(rec):
                        h = rec.get("h", {}).get("30m")
                        if not h:
                            return None
                        ups = rec.get("ups", 2)
                        # consensus: need a clear lean, else WAIT
                        if ups >= 3:
                            d = "UP"
                        elif ups <= 1:
                            d = "DOWN"
                        else:
                            d = None
                        return {
                            "time": rec.get("time"), "price": rec.get("price"),
                            "dir": d, "ups": ups,
                            "conf": h.get("conf"),
                            "tgt": h.get("tgt"), "sl": h.get("sl"),
                            "tgt_pts": h.get("tgt_pts"), "sl_pts": h.get("sl_pts"),
                            "strong": bool(rec.get("signal", {}).get("strong")),
                            "outcome": h.get("outcome"), "move": h.get("move"),
                        }

                    rows = [s for r in today if (s := single_of(r))]
                    scored = [s for s in rows
                              if s["dir"] and s["outcome"] in ("hit", "miss")]
                    hits = sum(1 for s in scored if s["outcome"] == "hit")
                    fce = FC_ENGINES.get(name)
                    out[name] = {
                        "current": single_of(st.get("mforecast") or {})
                                   if st.get("mforecast") else None,
                        "history": rows[-300:],
                        "hits": hits, "n": len(scored),
                        "pct": round(hits / len(scored) * 100) if scored else None,
                        "strong_acc": getattr(fce, "strong_acc", None) if fce else None,
                        "price": st.get("price"), "mode": st.get("mode"),
                    }
            self._json(out)
        elif url.path == "/api/pulse":
            # PULSE window: ONE call per minute (dir + entry + good-margin target/SL).
            # No time limit — a trade is scored on whether TARGET or STOPLOSS is hit
            # first (any time before close), not at a fixed horizon.
            #   flavor "sniper"  -> tight target + strict gates: fewest, fastest calls
            #   flavor "hit"     -> tight target: higher green-rate, small wins
            #   flavor "classic" -> same target, wider stop: fewer, cleaner trades
            qs = parse_qs(url.query)
            flavor = qs.get("mode", ["hit"])[0]
            #                 tgt_k, sl_k, re-entry-interval-min (0 = flips only, no re-entry)
            # NOTE: multipliers are x1.5 of the tested values because TGT_ATR moved
            # from 1.5 to 1.0 — `base` shrank by a third, so these keep Pulse's
            # actual point targets exactly where the backtests measured them.
            _FL = {"sniper":  (0.45, 2.4, 75),   # fast green ticks, very selective
                   "hit":     (0.75, 1.5, 45),   # first entry + rare late re-entry
                   "classic": (0.75, 3.0, 0)}    # quality: FIRST entry of a direction
                                                 # only. Re-entries into an ageing trend
                                                 # decay (NIFTY green 44%->38%->35%).
            TGT_K, SL_K, REENTRY_MIN = _FL.get(flavor, _FL["hit"])
            # Options theta kills a trade that drags. Backtested option P&L (expiry
            # theta, BankNifty): no stop -34.5, 45min -19.6, 30min -18.3, 15min -15.2.
            # Short is strictly better for an options trader — winners resolve fast
            # (33% inside 15 min), losers drag 110-215 min. 20 min lets the 12-16 min
            # winners complete while cutting the theta bleed.
            TIME_STOP = 20
            with LOCK:
                out = {}
                for name in nl.TICKERS:
                    st = STATE.get(name, {})
                    hist = FC_HIST.get(name, [])
                    day = st.get("day")
                    candles = list(st.get("candles", []))   # [t,o,h,l,c]
                    lv = st.get("live")   # in-progress candle -> real-time HIT/MISS
                    if lv and (not candles or lv[0] > candles[-1][0]):
                        candles = candles + [lv]
                    today = [r for r in hist if r.get("day") == day and "h" in r]

                    def one_call(rec):
                        h = rec.get("h", {}).get("30m")
                        h60 = rec.get("h", {}).get("60m")
                        if not h or not h60:
                            return None
                        ups = rec.get("ups", 2)
                        d = "UP" if ups >= 3 else "DOWN" if ups <= 1 else h["dir"]
                        entry = rec.get("price")
                        # BIGGER "swing" target (60m-sized) for high-profit trades
                        base = h60.get("tgt_pts") or (h.get("tgt_pts") or 0)
                        tgt_pts = max(round(base * TGT_K), 1)   # flavor geometry
                        sl_pts = max(round(base * SL_K), 1)
                        # COST FLOOR: kill only the junk targets (BankNifty ~14-20 pts)
                        # that brokerage/slippage would eat. ~0.08% of price.
                        if entry and tgt_pts < entry * 0.0008:
                            return None
                        if d == "UP":
                            tgt, sl = entry + tgt_pts, entry - sl_pts
                        else:
                            tgt, sl = entry - tgt_pts, entry + sl_pts
                        # ADAPTIVE TIME BUDGET: a target only deserves as much time as
                        # the current speed needs to cover it (time = distance / speed).
                        # Fast market + small target -> quick cut; bigger target earns
                        # more time. Clamped so nothing drags for hours (theta).
                        spd_f = speed_at.get(rec.get("time"), 0) if speed_at else 0
                        spd_pts = spd_f * entry if (spd_f and entry) else 0
                        # x3 buffer: (target/speed) is only the AVERAGE time — real moves
                        # stall and resume, so winners routinely need 2-3x that. Backtested
                        # (BN): pts/min +0.05 (x1) -> +0.34 (x2) -> +0.95 (x3), target-hit
                        # 27% -> 35%. Floor 30 min so a fast estimate can't cut a live move.
                        tstop = (tgt_pts / spd_pts) * 3.0 if spd_pts > 0 else 45
                        tstop = int(max(30, min(90, round(tstop / 5) * 5)))
                        sig = rec.get("signal", {})
                        qscore = 50 + (sig.get("align", 0) or 0) * 5
                        qscore += (h.get("conf") or 50) - 50
                        qscore += 10 if sig.get("ultra") else 0
                        qscore += 8 if sig.get("hi_conviction") else 0
                        qscore += 7 if sig.get("strong") else 0
                        qscore -= 14 if sig.get("window") == "midday" else 0
                        qscore = round(_clamp(qscore, 0, 99), 1)
                        return {"time": rec.get("time"), "price": entry, "dir": d,
                                "tstop": tstop,
                                "conf": h.get("conf"), "strong": bool(sig.get("strong")),
                                "ultra": bool(sig.get("ultra")),
                                "hi_conviction": bool(sig.get("hi_conviction")),
                                "acc": sig.get("acc"),
                                "quality": qscore,
                                "align": sig.get("align", 0),
                                "midday": sig.get("window") == "midday",
                                "tgt": round(tgt, 1), "sl": round(sl, 1),
                                "tgt_pts": tgt_pts, "sl_pts": sl_pts}

                    def tmin(t):
                        return int(t[:2]) * 60 + int(t[3:5])

                    def path_outcome(call):
                        # First-touch race between TARGET and STOPLOSS.
                        #   HIT     = target touched first  -> +tgt_pts, time-to-hit
                        #   MISS    = stoploss touched first -> -sl_pts
                        # If neither touched by close, judge by the final price:
                        #   PROFIT  = ended on the profit side (partial gain)
                        #   MISS    = ended on the loss side
                        d, tgt, sl, entry = call["dir"], call["tgt"], call["sl"], call["price"]
                        t0 = call["time"]; t0m = tmin(t0)
                        seen_after = False; last_c = entry; held = 0
                        for b in candles:
                            if b[0] <= t0:
                                continue
                            seen_after = True
                            held = tmin(b[0]) - t0m
                            hi, lo, last_c = b[2], b[3], b[4]
                            # 1) target / stoploss decided on THIS bar
                            ht = (d == "UP" and hi >= tgt) or (d == "DOWN" and lo <= tgt)
                            hs = (d == "UP" and lo <= sl) or (d == "DOWN" and hi >= sl)
                            if ht and hs:           # both in one bar -> conservative miss
                                return {"outcome": "miss", "mins": held,
                                        "pts": -call["sl_pts"]}
                            if ht:
                                return {"outcome": "hit", "mins": held,
                                        "pts": call["tgt_pts"]}
                            if hs:
                                return {"outcome": "miss", "mins": held,
                                        "pts": -call["sl_pts"]}
                            # 2) TIME STOP — options theta eats a trade that drags on.
                            # Budget is per-trade: target distance / current speed.
                            if held >= call.get("tstop", TIME_STOP):
                                move = (last_c - entry) if d == "UP" else (entry - last_c)
                                return {"outcome": "profit" if move > 0 else "miss",
                                        "mins": held, "pts": round(move, 1),
                                        "timed": True}
                        if not seen_after:
                            return {"outcome": "open", "mins": None, "pts": None}
                        move = (last_c - entry) if d == "UP" else (entry - last_c)
                        if move > 0:                 # in profit but target not reached
                            return {"outcome": "profit", "mins": held, "pts": round(move, 1)}
                        return {"outcome": "miss", "mins": held, "pts": round(move, 1)}

                    # SELECTIVITY GATE: STRONG + prime hours (open included), one
                    # trade per STRONG run. (Dropped the expansion filter — it was
                    # cutting genuine STRONG winners like the 9:24 +200 move for only
                    # a marginal expectancy gain.)
                    def prime_ok(t):
                        hm = t[:5]                       # include the open (backtested ~60%)
                        return ("09:19" < hm < "14:30") and not ("11:00" <= hm <= "13:30")
                    # Log a new distinct trade when direction flips, OR when 30+ min
                    # have passed in the same trend (a re-entry). Gives a handful of
                    # tagde trades on a trending day, stable & deterministic.
                    # Quick decisive flips (this produced the clean winning trades).
                    # MIN_GAP just prevents two trades stacking within 15 min.
                    # CHOP FILTER: in a sideways range the model flips LONG/SHORT at
                    # nearly the same price (no way to profit). Efficiency = net travel
                    # / total path over the last hour; low value = pure chop -> stay out.
                    CHOP_MIN = 0.18 if flavor == "sniper" else 0.12
                    # only the worst sideways grind is skipped outside sniper mode
                    # SPEED GATE — the single best predictor of a fast, high-velocity
                    # move. Backtested: setups in the top-33% of recent speed reach
                    # >=5 pts/min 68% of the time (vs 39% for the slowest), and when
                    # they work they average ~91 points in ~8 minutes. Threshold is
                    # price-normalised so it works on both indices.
                    SPEED_MIN = 0.000175 if flavor == "sniper" else 0.0001414
                    MOM_MIN = 0.0021 if flavor == "sniper" else 0.0017
                    eff_at = {}; speed_at = {}; mom_at = {}
                    if len(candles) > 70:
                        cl = pd.Series([b[4] for b in candles],
                                       index=[b[0] for b in candles])
                        net = (cl - cl.shift(60)).abs()
                        path = cl.diff().abs().rolling(60).sum()
                        e = (net / path).replace([np.inf, -np.inf], np.nan)
                        eff_at = {t: v for t, v in e.items() if pd.notna(v)}
                        spd = cl.diff().abs().rolling(30).mean() / cl   # frac per min
                        speed_at = {t: v for t, v in spd.items() if pd.notna(v)}
                        mo = (cl.pct_change(30)).abs()                  # 30-min thrust
                        mom_at = {t: v for t, v in mo.items() if pd.notna(v)}

                    # QUIET GATE (BankNifty only). The vol-LSTM (~68% OOS) says whether
                    # the next 30 min is a BIG move or QUIET. Directional trades taken
                    # on QUIET stretches are clearly worse: over 351 days green 45.4%
                    # vs 51.8%, and skipping them cut total loss by 35%. It is the one
                    # filter that held up in BOTH halves of the sample (51.8% / 51.9%),
                    # which is why it is here and the others were dropped.
                    # ---- META FILTER (BankNifty): a second model that scores each
                    # setup on "will this earn >=5 pts/min?". On 106 unseen days it
                    # lifted the gated feed from +1.44 to +1.82 pts/min and cut the
                    # total loss by 74%. Fails open if anything is missing.
                    _mf = META_FILTERS.get(name)
                    _m5 = _mind = None
                    if _mf is not None and _mf.ok and len(candles) > 10:
                        try:
                            from oracle.models import meta_filter as _mfm
                            veng0 = VOL_ENGINES.get(name)
                            warm = getattr(veng0, "warmup5", None) if veng0 else None
                            df1 = pd.DataFrame(candles,
                                               columns=["t", "Open", "High", "Low", "Close"])
                            df1.index = pd.to_datetime(day + " " + df1["t"])
                            f5 = df1.resample("5min").agg(
                                {"Open": "first", "High": "max",
                                 "Low": "min", "Close": "last"}).dropna()
                            if warm is not None and len(warm):
                                f5 = pd.concat([warm, f5])
                                f5 = f5[~f5.index.duplicated(keep="last")].sort_index()
                            _m5 = f5
                            _mind = _mfm.indicators(f5)
                        except Exception:
                            _m5 = _mind = None

                    _qcache = {}


                    def meta_prob(c):
                        """Second model's score for one setup, or None if unscorable."""
                        if _mf is None or not _mf.ok or _m5 is None:
                            return None
                        try:
                            from oracle.models import meta_filter as _mfm
                            stamp = pd.to_datetime(f"{day} {c['time']}")
                            pos = _m5.index.searchsorted(stamp, side="right") - 1
                            if pos < 30 or pos >= len(_m5):
                                return None
                            pv = vol_at(c["time"])
                            if not pv:
                                return None
                            pbig = (pv["prob"] / 100.0 if pv["vol"] == "BIG"
                                    else 1 - pv["prob"] / 100.0)
                            side = 1 if c["dir"] == "UP" else -1
                            day_start = _m5.index.searchsorted(
                                pd.to_datetime(f"{day} 09:15"), side="left")
                            row = _mfm.features_at(_m5, _mind, pos, side, pbig,
                                                   c["tgt_pts"], c.get("tstop", 45),
                                                   max(pos - day_start, 0))
                            return _mf.prob(row)
                        except Exception:
                            return None

                    def meta_ok(c):
                        """True = act on it."""
                        if _mf is None or not _mf.ok or _m5 is None:
                            return True            # no model here -> gate doesn't apply
                        p = meta_prob(c)
                        if p is None:
                            return False           # can't judge it -> don't take it
                        return p >= _mf.thr

                    def vol_at(t):
                        """Vol-LSTM reading AS OF that minute (never with hindsight)."""
                        veng = VOL_ENGINES.get(name)
                        if not veng or not day:
                            return None
                        key = t[:4]                      # cache per 10-min block
                        if key in _qcache:
                            return _qcache[key]
                        try:
                            upto = [b for b in candles if b[0] <= t]
                            pv = veng.predict(upto, day) if len(upto) > 5 else None
                        except Exception:
                            pv = None
                        _qcache[key] = pv
                        return pv

                    def is_quiet(t):
                        if name != "BANKNIFTY":
                            return False
                        pv = vol_at(t)
                        return bool(pv and pv.get("vol") == "QUIET")

                    MIN_GAP = 15
                    calls = []
                    pos_dir = None       # direction of the LAST LOGGED trade (not last seen
                                         # signal) — a gap in STRONG must not fake a "flip"
                    last_t = None
                    for r in today:
                        c = one_call(r)
                        if not c:
                            continue
                        ok = c["strong"] and prime_ok(c["time"])
                        # High hit-rate mode ALSO shows midday signals (flagged, weaker)
                        if not ok and flavor == "hit" and c["midday"] and c["align"] >= 3:
                            ok = True
                        if flavor == "sniper":
                            ok = (c["strong"] and prime_ok(c["time"])
                                  and c["align"] >= 3
                                  and (c["ultra"] or c["hi_conviction"]
                                       or c["quality"] >= 82))
                        if not ok:
                            continue
                        # sideways chop -> no directional call at all
                        if eff_at and eff_at.get(c["time"], 1.0) < CHOP_MIN:
                            continue
                        # SPEED GATE — skip slow markets: a target can only be reached
                        # quickly if price is actually moving quickly right now.
                        if speed_at and speed_at.get(c["time"], 9.0) < SPEED_MIN:
                            continue
                        # THRUST GATE — the move must already have real force behind it
                        if mom_at and mom_at.get(c["time"], 9.0) < MOM_MIN:
                            continue
                        # QUIET GATE — no directional trade when a quiet stretch is
                        # predicted (BankNifty; see note above)
                        if is_quiet(c["time"]):
                            continue
                        # META FILTER — second model scores this exact setup. Setups
                        # that fall short are held aside, not thrown away: on ~14% of
                        # days nothing clears the bar, and an empty feed tells you
                        # nothing. The best of those is shown afterwards, flagged.
                        # META FILTER — a setup that falls short is LABELLED, never
                        # dropped. Hiding them made the feed look like it was losing
                        # entries: BankNifty can go a whole day with nothing clearing
                        # the bar, so rows kept vanishing. Now every setup stays and
                        # the tag says which ones the quality model actually backs.
                        if not meta_ok(c):
                            c = dict(c, below_bar=True)
                        tm = tmin(c["time"])
                        flip = c["dir"] != pos_dir
                        reentry = (REENTRY_MIN > 0 and last_t is not None
                                   and tm - last_t >= REENTRY_MIN)
                        far = last_t is None or tm - last_t >= MIN_GAP
                        if (flip or reentry) and far:
                            calls.append(c)
                            last_t = tm
                            pos_dir = c["dir"]

                    # ---- CLOSING MOMENTUM (research-backed, independent signal) ----
                    # Gao, Han, Li & Zhou, "Market Intraday Momentum", J. Financial
                    # Economics 2018: the first half-hour return predicts the last
                    # half-hour return, and the effect is strongest when the opening
                    # move is large. Replicated on 350 days of this data:
                    #   BANKNIFTY |open 30m| > 0.38%  -> 57.1% win, +21.9 pts avg
                    #   NIFTY50   |open 30m| > 0.32%  -> 52.4% win, +10.5 pts avg
                    OPEN_THR = {"BANKNIFTY": 0.0038}.get(name, 0.0032)
                    try:
                        f0 = [b for b in candles if "09:15" <= b[0] <= "09:45"]
                        if f0 and any(b[0] >= "14:55" for b in candles):
                            r_open = f0[-1][4] / f0[0][1] - 1
                            if abs(r_open) > OPEN_THR:
                                ent = next(b for b in candles if b[0] >= "14:55")
                                px = ent[4]
                                d2 = "UP" if r_open > 0 else "DOWN"
                                tp = max(round(px * 0.0014), 1)
                                calls.append({
                                    "time": ent[0], "price": px, "dir": d2,
                                    "conf": 57 if name == "BANKNIFTY" else 52,
                                    "strong": True, "align": 4, "midday": False,
                                    "tstop": 30, "paper": True,
                                    "tgt": round(px + (tp if d2 == "UP" else -tp), 1),
                                    "sl": round(px - (tp if d2 == "UP" else -tp), 1),
                                    "tgt_pts": tp, "sl_pts": tp})
                    except Exception:
                        pass

                    calls.sort(key=lambda c: c["time"])
                    for c in calls:
                        c.update(path_outcome(c))
                    resolved = [c for c in calls if c["outcome"] != "open"]
                    hits = [c for c in resolved if c["outcome"] == "hit"]
                    profits = [c for c in resolved if c["outcome"] == "profit"]
                    greens = hits + profits
                    misses = [c for c in resolved if c["outcome"] == "miss"]
                    hit_mins = [c["mins"] for c in hits if c["mins"] is not None]
                    green_mins = [c["mins"] for c in greens if c["mins"] is not None]
                    net_pts = round(sum(c.get("pts") or 0 for c in resolved), 1)
                    out[name] = {
                        "current": calls[-1] if calls else None,
                        "history": calls[-250:],
                        "hits": len(hits), "profits": len(profits),
                        "greens": len(greens),
                        "misses": len(misses), "n": len(resolved),
                        "pct": round(len(greens) / len(resolved) * 100) if resolved else None,
                        "hit_pct": round(len(hits) / len(resolved) * 100) if resolved else None,
                        "net_pts": net_pts,
                        "avg_hit_min": round(sum(hit_mins) / len(hit_mins)) if hit_mins else None,
                        "avg_green_min": round(sum(green_mins) / len(green_mins))
                                         if green_mins else None,
                        "avg_hit_pts": round(sum(c["pts"] for c in hits) / len(hits))
                                       if hits else None,
                        "open": sum(1 for c in calls if c["outcome"] == "open"),
                        "price": st.get("price"), "mode": st.get("mode"),
                        "flavor": flavor, "tgt_k": TGT_K, "sl_k": SL_K,
                    }
            self._json(out)
        elif url.path == "/api/options":
            # Options strategy engine — vol forecast -> straddle / iron condor
            import options_strategy, datetime as _dt
            with LOCK:
                vix = VIX_TICKS[-1][1] if VIX_TICKS else 13.0
                st_all = {n: dict(STATE.get(n, {})) for n in nl.TICKERS}
                vengines = {n: VOL_ENGINES.get(n) for n in nl.TICKERS}
            # days to next Thursday (weekly expiry proxy), min 1
            today = _dt.date.today()
            dte = ((3 - today.weekday()) % 7) or 7
            out = {}
            for name in nl.TICKERS:
                st = st_all[name]
                spot = st.get("price")
                eng = vengines[name]
                pred = None
                if eng and st.get("candles") and st.get("day"):
                    pred = eng.predict(list(st["candles"]), st["day"])
                regime = (pred or {}).get("vol", "QUIET")
                rec = options_strategy.recommend(
                    name, spot, vix, dte, regime,
                    capital=CONFIG["capital"], risk_pct=2.0) if spot else None
                out[name] = {"rec": rec, "vol_pred": pred, "vix": round(vix, 1),
                             "dte": dte}
            self._json(out)
        elif url.path == "/api/smartmoney":
            try:
                from oracle.strategy import smart_money
                self._json(smart_money.bias())
            except Exception as exc:
                self._json({"error": str(exc)[:80]})
        elif url.path == "/api/volatility":
            # Big-move (volatility) forecast — direction-FREE, ~68-74% OOS.
            with LOCK:
                out = {}
                for name in nl.TICKERS:
                    st = STATE.get(name, {})
                    candles = list(st.get("candles", []))
                    day = st.get("day")
                    eng = VOL_ENGINES.get(name)
                    pred = eng.predict(candles, day) if eng and day else None
                    out[name] = {
                        "pred": pred, "price": st.get("price"),
                        "mode": st.get("mode"),
                        "acc": getattr(eng, "acc", None) if eng else None,
                    }
            self._json(out)
        elif url.path == "/api/ai":
            force = q.get("force", ["0"])[0] == "1"
            text, err = ai_analyst.briefing(ai_snapshot(), force=force)
            if text:
                log_ai("analyst", {"text": text})
            self._json({"text": text, "error": err})
        elif url.path == "/api/copilot":
            force = q.get("force", ["0"])[0] == "1"
            verdict, err = ai_analyst.copilot(ai_snapshot(), force=force)
            if verdict:
                log_ai("copilot", verdict)
            self._json({"verdict": verdict, "error": err})
        elif url.path == "/api/tick":
            with LOCK:
                snap = {k: {"price": v["price"], "live": v.get("live")}
                        for k, v in STATE.items()}
            self._json(snap)
        elif url.path == "/api/state":
            with LOCK:
                # only YOUR trades (manual) — auto-trader P&L is excluded
                realized = sum(t["pnl"] for s in STATE.values()
                               for t in s["trades"] if t.get("kind") == "MANUAL")
                unreal = 0.0
                for s in STATE.values():
                    for key in ("auto_pos", "manual_pos"):
                        p = s[key]
                        if p and s["price"]:
                            d = 1 if (p.get("side") == "LONG" or p.get("dir") == 1) else -1
                            unreal += d * (s["price"] - p["entry"]) * p["qty"]
                    s["manual_pos_view"] = serialize_pos(s["manual_pos"]) \
                        if s["manual_pos"] else None
                snap = {"capital": CONFIG["capital"], "risk": CONFIG["risk"],
                        "realized": round(realized), "unrealized": round(unreal),
                        "indices": {k: {kk: vv for kk, vv in v.items()
                                        if kk != "manual_pos"}
                                    for k, v in STATE.items()}}
            self._json(snap)
        elif url.path == "/api/manual":
            name = q.get("index", [""])[0]
            action = q.get("action", [""])[0]
            raw_qty = q.get("qty", [""])[0]
            with LOCK:
                st = STATE.get(name)
                if not st or st["price"] is None:
                    return self._json({"ok": False, "msg": "no price yet"})
                price = st["price"]
                mp = st["manual_pos"]
                if action == "close" and mp:
                    d = mp["dir"]
                    pnl = d * (price - mp["entry"]) * mp["qty"]
                    r = pnl / (abs(mp["entry"] - mp["sl"]) * mp["qty"])
                    record(st, "MANUAL", "LONG" if d == 1 else "SHORT",
                           mp["entry"], price, mp["qty"], r, pnl, "manual close",
                           mp.get("last_ts", ""))
                    st["manual_pos"] = None
                    return self._json({"ok": True, "msg": f"closed at {price}"})
                if action in ("buy", "sell") and not mp:
                    if st["finished"]:
                        return self._json({"ok": False, "msg": "session over"})
                    d = 1 if action == "buy" else -1
                    sl_pts = price * MANUAL_SL_PCT
                    qty = CONFIG["capital"] * CONFIG["risk"] / 100 / sl_pts
                    if raw_qty:
                        try: qty = float(raw_qty)
                        except: pass
                    st["manual_pos"] = {"dir": d, "entry": price,
                                        "sl": price - d * sl_pts,
                                        "tgt": price + d * 1.5 * sl_pts,
                                        "qty": qty, "last_ts": ""}
                    return self._json({"ok": True,
                                       "msg": f"{action.upper()} {qty:.1f}u @ {price}"})
                return self._json({"ok": False,
                                   "msg": "already in a trade" if mp else "nothing to close"})
        elif url.path == "/api/set_qty":
            name = q.get("index", [""])[0]
            raw_qty = q.get("qty", [""])[0]
            with LOCK:
                st = STATE.get(name)
                if st and "eng" in st:
                    # In replay mode 'eng' isn't stored in state, wait, eng is not in state!
                    # Let's store user_qty in state directly
                    st["user_qty"] = raw_qty
            return self._json({"ok": True, "msg": "updated"})
        else:
            self._json({"ok": False}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=100000)
    ap.add_argument("--risk", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8181)
    args = ap.parse_args()
    CONFIG["capital"], CONFIG["risk"] = args.capital, args.risk

    global FEED
    if os.environ.get("NO_ANGEL") == "1":
        FEED, feed_msg = None, "Angel skipped (NO_ANGEL=1) — using Yahoo (delayed but reliable)."
        print(feed_msg)
    else:
        FEED, feed_msg = angel_data.connect()
    if FEED is None:
        feed_msg = "Angel One not ready — using Yahoo (delayed but reliable)."
    if FEED is not None and hasattr(FEED, "on_tick"):
        FEED.on_tick = on_stream_tick
    print(feed_msg)
    live = market_open_now()
    print(f"Mode: {'LIVE (market open)' if live else 'CLOSED (showing last real session)'}")
    print("Training models + fetching data… first candles appear in ~20 seconds.")
    worker = live_worker if live else closed_worker
    for name, symbol in nl.TICKERS.items():
        STATE[name] = new_state(name)
        threading.Thread(target=worker, args=(name, symbol), daemon=True).start()
        if H2H:                    # opt-in head-to-head logger, off the feed path
            threading.Thread(target=h2h_logger, args=(name,), daemon=True).start()

    url = f"http://localhost:{args.port}"
    print(f"Dashboard: {url}   (Ctrl+C to stop)")
    threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
