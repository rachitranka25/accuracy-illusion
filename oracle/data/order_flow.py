#!/usr/bin/env python3
"""
OFI collector (low-latency, multi-level) — builds your own Order-Flow-Imbalance
dataset for free from Angel's live best-5 futures depth.

NSE/SEBI don't give historical order-book data free (only expensive paid feeds),
so we collect it live going forward. Index spot has no order book (not traded) —
we subscribe to the NIFTY/BANKNIFTY *futures*.

Low-latency design (this matters here — OFI alpha decays in milliseconds):
  * the on_data callback does only O(5) work and appends to a bounded deque
    (ring buffer). No heavy work on the hot path.
  * per-minute aggregation happens in a separate flusher thread.

Richer features (per microstructure research):
  OFI-touch   = level-1 imbalance                     (bid_q1 - ask_q1)/(sum)
  OFI-5       = 5-level imbalance
  OFI-weight  = depth-weighted (nearer levels weigh more) — most predictive
  microprice  = (bid1*ask_q1 + ask1*bid_q1)/(bid_q1+ask_q1), vs mid
  spread      = ask1 - bid1

Output: data_cache/{name}_ofi.csv (per-minute averages).
Run during market hours (Mon-Fri 9:15-15:30):  python3 ofi_collector.py
"""
import warnings; warnings.filterwarnings("ignore")
import json, time, threading, datetime as dt, urllib.request
from pathlib import Path
from collections import deque, defaultdict
try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    IST = None

from oracle.data import angel as angel_data
from oracle import paths

CACHE = paths.CACHE
MASTER = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
LEVEL_W = (1.0, 0.6, 0.35, 0.2, 0.1)          # depth weights (nearer = heavier)


def front_future_tokens():
    data = json.load(urllib.request.urlopen(MASTER, timeout=40))
    out = {}
    for idx in ("NIFTY", "BANKNIFTY"):
        futs = [d for d in data if d.get("name") == idx
                and d.get("instrumenttype") == "FUTIDX"
                and d.get("exch_seg") == "NFO"]

        def ed(d):
            try:
                return dt.datetime.strptime(d["expiry"], "%d%b%Y")
            except Exception:
                return dt.datetime.max
        futs.sort(key=ed)
        if futs:
            out[idx] = str(futs[0]["token"])
    return out


def depth_sides(msg):
    for bk, ak in (("best_5_buy_data", "best_5_sell_data"),
                   ("best5buydata", "best5selldata"),
                   ("bestBids", "bestAsks")):
        if bk in msg and ak in msg:
            return msg[bk], msg[ak]
    return None, None


def _f(row, *keys):
    for k in keys:
        if k in row:
            try:
                return float(row[k])
            except Exception:
                return 0.0
    return 0.0


class OFICollector:
    def __init__(self):
        # ring buffers (bounded) — O(1) append, no unbounded growth on hot path
        self.buf = defaultdict(lambda: deque(maxlen=4000))
        self.lock = threading.Lock()
        self.tok2name = {}

    def run(self):
        print("logging in to Angel…")
        feed, msg = angel_data.connect()
        if feed is None:
            print("Angel connect failed:", msg); return
        api = feed.api
        toks = front_future_tokens()
        self.tok2name = {v: k for k, v in toks.items()}
        print("collecting OFI (low-latency, multi-level) for:", toks)

        from SmartApi.smartWebSocketV2 import SmartWebSocketV2
        jwt = feed._session["data"]["jwtToken"]
        ft = api.getfeedToken()
        sws = SmartWebSocketV2(jwt, feed.cfg["api_key"], feed.cfg["client_code"], ft)

        def on_open(w):
            sws.subscribe("ofi", 3, [{"exchangeType": 2,
                                      "tokens": list(toks.values())}])
            print("subscribed (SNAP_QUOTE depth). Waiting for market ticks…")

        seen_raw = [False]

        def on_data(w, m):
            # ---- HOT PATH: keep this minimal & O(5). Just compute + append. ----
            try:
                if isinstance(m, (bytes, str)):
                    return
                tok = str(m.get("token", "")).strip('"')
                name = self.tok2name.get(tok)
                if not name:
                    return
                bids, asks = depth_sides(m)
                if not bids or not asks:
                    if not seen_raw[0]:
                        seen_raw[0] = True
                        print("  (depth keys seen:", list(m.keys()), ")")
                    return
                # single O(5) pass over levels
                bq = aq = wbq = waq = 0.0
                for i in range(min(5, len(bids), len(asks))):
                    q_b = _f(bids[i], "quantity", "qty", "no of quantity")
                    q_a = _f(asks[i], "quantity", "qty", "no of quantity")
                    w = LEVEL_W[i]
                    bq += q_b; aq += q_a
                    wbq += w * q_b; waq += w * q_a
                b1q = _f(bids[0], "quantity", "qty", "no of quantity")
                a1q = _f(asks[0], "quantity", "qty", "no of quantity")
                b1p = _f(bids[0], "price", "Price")
                a1p = _f(asks[0], "price", "Price")
                if bq + aq <= 0 or b1q + a1q <= 0:
                    return
                ofi_touch = (b1q - a1q) / (b1q + a1q)
                ofi5 = (bq - aq) / (bq + aq)
                ofi_w = (wbq - waq) / (wbq + waq) if (wbq + waq) else 0.0
                mid = (b1p + a1p) / 2 if (b1p and a1p) else 0.0
                micro = ((b1p * a1q + a1p * b1q) / (b1q + a1q)) if (b1p and a1p) else 0.0
                micro_dev = (micro - mid) / mid if mid else 0.0
                spread = (a1p - b1p) if (a1p and b1p) else 0.0
                self.buf[name].append((ofi_touch, ofi5, ofi_w, micro_dev, spread))
            except Exception:
                pass

        sws.on_open = on_open
        sws.on_data = on_data
        sws.on_error = lambda w, e: None
        sws.on_close = lambda w, *a: None
        threading.Thread(target=self._flusher, daemon=True).start()
        sws.connect()

    def _flusher(self):
        while True:
            now = dt.datetime.now()
            time.sleep(60 - now.second)
            stamp = dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M")
            with self.lock:
                snap = {n: list(b) for n, b in self.buf.items()}
                for b in self.buf.values():
                    b.clear()
            for name, vals in snap.items():
                if not vals:
                    continue
                k = len(vals)
                cols = list(zip(*vals))     # transpose -> per-metric lists
                avg = [sum(c) / k for c in cols]
                f = CACHE / f"{name}_ofi.csv"
                new = not f.exists()
                with open(f, "a") as fh:
                    if new:
                        fh.write("time,ofi_touch,ofi5,ofi_weighted,microprice_dev,spread,snapshots\n")
                    fh.write(f"{stamp},{avg[0]:.4f},{avg[1]:.4f},{avg[2]:.4f},"
                             f"{avg[3]:.6f},{avg[4]:.2f},{k}\n")
                print(f"  {stamp}  {name}: OFI-w {avg[2]:+.3f} · touch {avg[0]:+.3f} "
                      f"· spread {avg[4]:.1f} ({k} snaps)")


if __name__ == "__main__":
    OFICollector().run()
