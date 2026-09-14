#!/usr/bin/env python3
"""
Angel One SmartAPI data feed for the NIFTY BEAST.

Reads credentials from angel_config.json (same folder). If the file is
incomplete or login fails, callers fall back to free Yahoo data — the
dashboard keeps working either way.

angel_config.json needs:
  api_key     - from your SmartAPI app (you have this)
  client_code - your Angel One login ID (like A123456)
  pin         - your Angel One login PIN
  totp_secret - from https://smartapi.angelbroking.com/enable-totp
                (the long letters-code shown under the QR; NOT the 6-digit
                 number, which changes every 30 seconds)

NEVER share this file or its contents with anyone.
"""

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from oracle import paths

CONFIG_PATH = paths.ROOT / "angel_config.json"
TZ = "Asia/Kolkata"

# NSE index tokens for the historical candle API
TOKENS = {"BANKNIFTY": ("99926009", "NSE"), "NIFTY50": ("99926000", "NSE"),
          "FINNIFTY": ("99926037", "NSE"),
          "INDIAVIX": ("99926017", "NSE")}   # VIX streams as a model feature
INTERVALS = {"1m": "ONE_MINUTE", "5m": "FIVE_MINUTE", "1d": "ONE_DAY"}


def load_config():
    # env vars (.env) first, angel_config.json fallback — see envconfig.py
    from oracle.config import from_env_or_json
    return from_env_or_json("angel_config.json", {
        "api_key": "ANGEL_API_KEY",
        "client_code": "ANGEL_CLIENT_CODE",
        "pin": "ANGEL_PIN",
        "totp_secret": "ANGEL_TOTP_SECRET",
    })


class AngelFeed:
    """Logged-in SmartAPI session with a candle fetcher. Thread-safe."""

    def __init__(self, cfg):
        # SmartAPI logs full request headers/bodies (incl. PIN + API key) on any
        # network error via logzero. Silence it so credentials never hit disk.
        try:
            import logzero, logging as _lg
            logzero.loglevel(_lg.CRITICAL)
        except Exception:
            pass
        from SmartApi import SmartConnect
        import pyotp
        self._lock = threading.Lock()
        self._last_call = 0.0          # SmartAPI rate limit: keep calls spaced out
        self.api = SmartConnect(api_key=cfg["api_key"])
        totp = pyotp.TOTP(cfg["totp_secret"]).now()
        session = self.api.generateSession(cfg["client_code"], cfg["pin"], totp)
        if not session or not session.get("status"):
            raise RuntimeError(f"Angel One login failed: {session}")
        self.client_code = cfg["client_code"]
        self.cfg = cfg
        self._session = session
        self._ws_prices = {}           # name -> (price, received_at)
        self.on_tick = None            # optional callback(name, price) per tick
        self._start_stream()           # push-based ticks, like Angel's own app

    def _start_stream(self):
        """Live tick stream via SmartWebSocketV2 (best effort; REST fallback)."""
        try:
            from SmartApi.smartWebSocketV2 import SmartWebSocketV2
            jwt = self._session["data"]["jwtToken"]
            feed_token = self.api.getfeedToken()
            sws = SmartWebSocketV2(jwt, self.cfg["api_key"],
                                   self.cfg["client_code"], feed_token)
            tok2name = {t: n for n, (t, _) in TOKENS.items()}

            def on_open(wsapp):
                sws.subscribe("beast", 1,        # mode 1 = LTP ticks
                              [{"exchangeType": 1, "tokens": list(tok2name)}])

            def on_data(wsapp, msg):
                try:
                    t = str(msg.get("token", "")).strip('"')
                    px = msg.get("last_traded_price")
                    if t in tok2name and px:
                        name2 = tok2name[t]
                        self._ws_prices[name2] = (px / 100.0, time.time())
                        cb = self.on_tick
                        if cb:
                            cb(name2, px / 100.0)
                except Exception:
                    pass

            sws.on_open = on_open
            sws.on_data = on_data
            sws.on_error = lambda *a: None
            sws.on_close = lambda *a: None
            threading.Thread(target=sws.connect, daemon=True).start()
        except Exception:
            pass                        # stream is a bonus; REST ltp still works

    def ltp(self, name):
        """Live price: streamed tick when fresh, REST call as fallback."""
        cached = self._ws_prices.get(name)
        if cached and time.time() - cached[1] < 15:
            return cached[0]
        token, exchange = TOKENS[name]
        with self._lock:
            gap = time.time() - self._last_call
            if gap < 0.35:
                time.sleep(0.35 - gap)
            resp = self.api.ltpData(exchange, name, token)
            self._last_call = time.time()
        if not resp or not resp.get("status") or not resp.get("data"):
            raise RuntimeError(f"ltp failed: {str(resp)[:80]}")
        return float(resp["data"]["ltp"])

    def candles(self, name, interval="1m", days=1):
        """Return an OHLC DataFrame (IST index) like the yfinance frames."""
        token, exchange = TOKENS[name]
        now = datetime.now()
        params = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": INTERVALS[interval],
            "fromdate": (now - timedelta(days=days)).strftime("%Y-%m-%d 09:15"),
            "todate": now.strftime("%Y-%m-%d %H:%M"),
        }
        with self._lock:                       # SmartAPI client is not thread-safe
            gap = time.time() - self._last_call
            if gap < 2.0:                      # stay well under the rate limit
                time.sleep(2.0 - gap)
            resp = self.api.getCandleData(params)
            self._last_call = time.time()
        if not resp or not resp.get("status") or not resp.get("data"):
            raise RuntimeError(f"candle fetch failed: {resp}")
        df = pd.DataFrame(resp["data"],
                          columns=["ts", "Open", "High", "Low", "Close", "Volume"])
        df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts"))).tz_convert(TZ)
        return df.astype(float)


def connect():
    """Return (feed, message). feed is None when not configured / login fails."""
    cfg = load_config()
    if cfg is None:
        return None, ("Angel One not configured — using free Yahoo data (delayed). "
                      "Fill angel_config.json to go real-time.")
    try:
        feed = AngelFeed(cfg)
        return feed, f"Angel One connected ✅ (client {feed.client_code}) — real-time data ON"
    except Exception as exc:
        return None, f"Angel One login failed ({exc}) — falling back to Yahoo data"


if __name__ == "__main__":
    feed, msg = connect()
    print(msg)
    if feed:
        for name in TOKENS:
            df = feed.candles(name, "1m", days=1)
            print(f"{name}: {len(df)} candles, last = {df.index[-1]} "
                  f"close {df['Close'].iloc[-1]:.1f}")
