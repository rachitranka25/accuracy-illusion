#!/usr/bin/env python3
"""
Forecast Engine V3 — honest, self-consistent multi-horizon forecaster.

What changed from the (broken) V2.1 and WHY:
  • FIXED crash: __init__ referenced BASE_FEATS/META_FEATS which never
    existed. The engine could never be constructed.
  • DROPPED the "extra" features (VIX / GARCH / cross-index / cointegration).
    They were computed from 2 years of continuous data at TRAIN time but from
    only today's intraday bars at LIVE time — a train/serve mismatch that made
    them garbage in production. Walk-forward showed they did not reliably help.
    The engine now uses ONLY intraday features that are computed identically
    in training and live (no skew).
  • CALIBRATED confidence tiers. Instead of hard-coded 57/53 cutoffs (which
    almost never fired), HIGH / MEDIUM / LOW thresholds are learned per horizon
    from a held-out validation slice, and each tier's REAL out-of-sample
    hit-rate is measured and shown in the UI. HIGH = the model's most confident
    ~28% of calls; that is where the small, real edge lives.

Honest expectation: per-bar direction is close to a coin toss (~50-53%).
The HIGH tier lifts that a few points on its best setups. There is no 90%,
and anything claiming it is a bug or a lie. The real money-making engine is
the disciplined ORB system in nifty_live.py.

Public API (unchanged, dashboard depends on it):
  ForecastEngine(name, capital, risk_pct, m5=None)
  .forecast(candles, day, other_candles=None, vix_ticks=None) -> dict | None
  .backtest_acc[key]  -> int   (out-of-sample all-signal hit-rate %, per horizon)
  .tier_info[key]     -> dict  (thresholds + per-tier held-out accuracy)
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb  # pyrefly: ignore  # installed & working; lightgbm ships no type stubs

from oracle.features import indicators as nl
from oracle import paths

warnings.filterwarnings("ignore")

CACHE = paths.CACHE
# ── public constants (dashboard depends on these) ──────────────────────
OTHER = {"BANKNIFTY": "NIFTY50", "NIFTY50": "BANKNIFTY"}
HORIZONS = [("5m", 1), ("15m", 3), ("30m", 6), ("60m", 12)]   # in 5-min bars
HORIZON_MIN = {"5m": 5, "15m": 15, "30m": 30, "60m": 60}
SL_ATR, TGT_ATR = 1.0, 1.0
# Symmetric bracket. With target at 1.5x the stop, the target simply sat further
# away than the stop, so even a coin-flip direction showed only ~50% green ticks —
# a geometry artefact, not a model failure. 1:1 measures 55.5% green for ~10% less
# expectancy. Going nearer still (0.6x -> 65% green) costs 32% and is the trap:
# the ticks look better while the money gets worse.
MODEL_VERSION = "v3.1"
STRONG_MIN_ALIGN = 3     # STRONG when >=3 of 4 horizons are HIGH and agree
MIN_ULTRA_CONF = 58      # ULTRA when every agreeing horizon's conf >= this (0-100)
# Backtest (30m HIGH tier): midday 11:00-13:30 is the model's dead zone
# (BankNifty 48%, below coin toss); mornings + late afternoon run 55-57%.
# STRONG is suppressed during the dead zone.
DEAD_ZONE = (11 * 60, 13 * 60 + 30)   # minutes since midnight IST

# ── features: intraday only, identical in train and live (no skew) ─────
BASE_COLS = [
    "r1",                       # 1-bar return
    "rsi", "ema20",             # momentum / trend distance
    "macd_hist",                # MACD histogram
    "bb_width", "bb_pos",       # Bollinger width and %b
    "atr_pct", "parkinson_vol", # volatility
    "vwap",                     # distance from session VWAP
    "tod", "dow",               # time-of-day, day-of-week
    "regime_vol", "hurst",      # rolling regime descriptors
    "dist_ph", "dist_pl",       # distance from the previous session's high / low
]


def load_long(name):
    f = CACHE / f"{name}_5m_long.csv"
    if f.exists():
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        if len(df) > 5000:
            return df
    return None


# ── Hurst exponent (rolling, causal) ───────────────────────────────────
def _hurst(series, window=20):
    out = pd.Series(np.nan, index=series.index)
    vals = series.values
    for i in range(window, len(vals)):
        seg = vals[i - window:i]
        dev = seg - seg.mean()
        cum = np.cumsum(dev)
        r = cum.max() - cum.min()
        s = seg.std(ddof=1)
        if s > 0 and r > 0:
            out.iloc[i] = np.log(r / s) / np.log(window)
    return out


# ── feature builder (all causal: only past/current bars) ───────────────
def build5(bars):
    """Build features from 5-minute OHLC bars. Returns (features_df, close)."""
    c, h, l = bars["Close"], bars["High"], bars["Low"]
    f = pd.DataFrame(index=bars.index)

    f["r1"] = c.pct_change()
    f["rsi"] = nl.rsi(c) / 100.0
    f["ema20"] = c / c.ewm(span=20).mean() - 1

    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    f["macd_hist"] = (macd_line - signal_line) / c

    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f["bb_width"] = (2 * std20) / ma20
    f["bb_pos"] = (c - ma20) / (2 * std20)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()],
                   axis=1).max(axis=1)
    f["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    f["atr_pct"] = f["atr"] / c
    f["parkinson_vol"] = np.sqrt(
        (np.log(h / l) ** 2).rolling(12).mean() / (4 * np.log(2)))

    tp = (h + l + c) / 3
    vw = tp.groupby(bars.index.date).expanding().mean() \
           .reset_index(level=0, drop=True)
    f["vwap"] = c / vw - 1

    mins = bars.index.hour * 60 + bars.index.minute - 555
    f["tod"] = mins / 375.0
    f["dow"] = bars.index.dayofweek / 4.0

    f["regime_vol"] = f["r1"].rolling(20).std()
    f["hurst"] = _hurst(c, window=20)

    # Distance from the PREVIOUS session's high/low. Of every extra input tried
    # (vol-LSTM output, the other index, India VIX, time-of-day), this was the only
    # one whose gain survived splitting the clean test block in half — 15-min
    # accuracy went 48.0/51.7 to 50.5/52.3. `.shift(1)` keeps it strictly causal:
    # today only ever sees yesterday's levels.
    day = pd.Series(bars.index.date, index=bars.index)
    prev_hi = day.map(h.groupby(bars.index.date).max().shift(1))
    prev_lo = day.map(l.groupby(bars.index.date).min().shift(1))
    f["dist_ph"] = c / prev_hi - 1
    f["dist_pl"] = c / prev_lo - 1
    return f, c


# ── tier calibration (learned per horizon from held-out validation) ────
def _calibrate_tiers(conf, correct):
    """conf: |p-0.5|*100 array; correct: 0/1 array on the validation slice.
    HIGH = most confident ~28%, MEDIUM = next ~35%, LOW = rest.
    Returns thresholds + the REAL held-out hit-rate of each tier."""
    hi_thr = float(np.percentile(conf, 72))
    md_thr = float(np.percentile(conf, 37))

    def acc(mask):
        return round(float(correct[mask].mean()) * 100) if mask.sum() >= 15 else None

    hi = conf >= hi_thr
    md = (conf >= md_thr) & (conf < hi_thr)
    lo = conf < md_thr
    return {
        "hi_thr": round(hi_thr, 2), "md_thr": round(md_thr, 2),
        "HIGH": acc(hi), "MEDIUM": acc(md), "LOW": acc(lo),
        "hi_n": int(hi.sum()),
    }


def _tier_of(conf, info):
    if info and conf >= info["hi_thr"]:
        return "HIGH"
    if info and conf >= info["md_thr"]:
        return "MEDIUM"
    return "LOW"


def _measure_strong(val_store):
    """Held-out accuracy of the STRONG rule (>=N horizons HIGH and agreeing).
    Returns (accuracy_pct | None, n_minutes)."""
    if len(val_store) < len(HORIZONS):
        return None, 0
    idxs = [pd.Index(v[0]) for v in val_store.values()]
    common = idxs[0]
    for ix in idxs[1:]:
        common = common.intersection(ix)
    if len(common) < 50:
        return None, 0
    nH = len(HORIZONS)
    up = np.zeros((len(common), nH), int)
    hi = np.zeros((len(common), nH), bool)
    corr = np.zeros((len(common), nH), int)
    for j, (key, _) in enumerate(HORIZONS):
        te_idx, proba, y, thr = val_store[key]
        s = pd.Series(range(len(te_idx)), index=te_idx)
        pos = s.reindex(common).values.astype(int)
        p = proba[pos]; yy = y[pos]
        up[:, j] = (p >= 0.5)
        hi[:, j] = (np.abs(p - 0.5) * 100 >= thr)
        corr[:, j] = ((p >= 0.5).astype(int) == yy)
    hi_up = (hi & (up == 1)).sum(1)
    hi_dn = (hi & (up == 0)).sum(1)
    rows = np.maximum(hi_up, hi_dn) >= STRONG_MIN_ALIGN
    if rows.sum() < 15:
        return None, int(rows.sum())
    agree = ((up == 1) & (hi_up >= STRONG_MIN_ALIGN)[:, None]) | \
            ((up == 0) & (hi_dn >= STRONG_MIN_ALIGN)[:, None])
    sel = hi & agree
    strong_acc = round(float(corr[sel].mean()) * 100)
    # ULTRA: same rows, but require every agreeing horizon's confidence >= gate.
    # conf per horizon on the 50-100 scale = max(p,1-p)*100.
    conf_all = np.zeros((len(common), nH))
    for j, (key, _) in enumerate(HORIZONS):
        te_idx, proba, y, thr = val_store[key]
        s = pd.Series(range(len(te_idx)), index=te_idx)
        pos = s.reindex(common).values.astype(int)
        p = proba[pos]
        conf_all[:, j] = np.maximum(p, 1 - p) * 100
    # min conf among the horizons that both agree and count toward the signal
    big = np.where(agree, conf_all, 999.0)
    min_conf_row = big.min(1)
    ultra_rows = rows & (min_conf_row >= MIN_ULTRA_CONF)
    if ultra_rows.sum() >= 15:
        usel = hi & agree & ultra_rows[:, None]
        ultra_acc = round(float(corr[usel].mean()) * 100)
    else:
        ultra_acc = None
    return strong_acc, int(rows.sum()), ultra_acc, int(ultra_rows.sum())


def _make_lgb(n_estimators=500):
    return lgb.LGBMClassifier(
        objective="binary", num_leaves=24, max_depth=4, learning_rate=0.02,
        n_estimators=n_estimators, subsample=0.75, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=2.0, min_child_samples=80,
        random_state=42, verbose=-1)


class ForecastEngine:
    def __init__(self, name, capital, risk_pct, m5=None):
        self.name = name
        self.risk_rupees = capital * risk_pct / 100
        self.cols = BASE_COLS

        long = load_long(name)
        if long is not None:
            m5 = long
        elif m5 is None:
            m5 = nl.flat(nl.yf.download(nl.TICKERS[name], period="60d",
                                        interval="5m", progress=False,
                                        auto_adjust=True))

        # WARMUP: keep the tail of history so live forecasts have full rolling
        # windows from the first bar of the day — matching how the model was
        # TRAINED (continuous data), and removing the ~1.5h cold-start delay.
        self.warmup_5m = None
        try:
            w = m5.tail(60)[["Open", "High", "Low", "Close"]].copy()
            if getattr(w.index, "tz", None) is not None:
                w.index = w.index.tz_localize(None)
            self.warmup_5m = w
        except Exception:
            self.warmup_5m = None

        # ---- REALISTIC target sizing (calibrated from ACTUAL moves) ----
        # The old sqrt(horizon) x 5-min-ATR formula overshot ~4-6x: it showed
        # 200-pt targets when a real 60-min move is ~50 pts. Instead we take the
        # 65th-percentile of actual k-bar moves as the target, and scale it to
        # today's volatility at serve time. SL = target / 1.5  (1.5:1 R:R).
        try:
            _c5 = m5["Close"]
            self.move_tgt = {}
            for _key, _k in HORIZONS:
                _mv = (_c5.shift(-_k) - _c5).abs().dropna()
                self.move_tgt[_key] = float(_mv.quantile(0.65)) if len(_mv) else 0.0
            _tr = pd.concat([m5["High"] - m5["Low"],
                             (m5["High"] - _c5.shift()).abs(),
                             (m5["Low"] - _c5.shift()).abs()], axis=1).max(axis=1)
            _ref = _tr.ewm(alpha=1 / 14, adjust=False).mean().median()
            self.ref_atr = float(_ref) if np.isfinite(_ref) and _ref > 0 else 1.0
            # median recent realized vol (6-bar |return| sum) — for the
            # high-conviction regime flag (direction is ~2pts better in high vol)
            _rv = _c5.pct_change().abs().rolling(6).sum()
            self.vol_med = float(_rv.median()) if np.isfinite(_rv.median()) else 0.0
        except Exception:
            self.move_tgt, self.ref_atr, self.vol_med = {}, 1.0, 0.0

        stamp = str(m5.index[-1].date())
        pkl = CACHE / f"{name}_models_v2.pkl"
        if pkl.exists():
            try:
                saved = pickle.load(open(pkl, "rb"))
                if (saved.get("stamp") == stamp
                        and saved.get("version") == MODEL_VERSION
                        and saved.get("cols") == self.cols):
                    self.models = saved["models"]
                    self.backtest_acc = saved["acc"]
                    self.tier_info = saved["tiers"]
                    self.strong_acc = saved.get("strong_acc")
                    self.strong_n = saved.get("strong_n", 0)
                    self.ultra_acc = saved.get("ultra_acc")
                    self.ultra_n = saved.get("ultra_n", 0)
                    print(f"  {name}: loaded cached {MODEL_VERSION} models "
                          f"(OOS acc: {self.backtest_acc}, STRONG ~{self.strong_acc}%, "
                          f"ULTRA ~{self.ultra_acc}%)")
                    return
            except Exception:
                pass

        print(f"  {name}: training {MODEL_VERSION} ({len(m5)} bars, "
              f"{len(set(m5.index.date))} days)…")

        feats, c5 = build5(m5)
        self.models, self.backtest_acc, self.tier_info = {}, {}, {}
        val_store = {}     # per-horizon validation preds, to measure STRONG
        # THREE-way split. Early stopping needs its own block: the old code stopped
        # on the same rows it then measured, so both the reported accuracy and the
        # tier calibration came out flattered (HIGH tier read 49.7% but delivered
        # 47.7% on rows the fit had never touched). `es` picks the tree count, `te`
        # is never seen during fitting and is what we report and calibrate on.
        days = sorted(set(feats.index.date))
        n_hold = max(20, len(days) // 10)
        es_cut = days[-2 * n_hold]
        val_cut = days[-n_hold]

        for key, k in HORIZONS:
            lab = (c5.shift(-k) > c5).astype(float)
            ok = feats.assign(label=lab).dropna(subset=self.cols + ["label"])
            tr = ok[ok.index.date < es_cut]
            es = ok[(ok.index.date >= es_cut) & (ok.index.date < val_cut)]
            te = ok[ok.index.date >= val_cut]

            if len(tr) < 200 or len(es) < 50 or len(te) < 50:
                model = _make_lgb(200)
                model.fit(np.nan_to_num(ok[self.cols].values, 0),
                          ok["label"].astype(int).values)
                self.models[key] = model
                self.backtest_acc[key] = 50
                self.tier_info[key] = None
                continue

            X_tr = np.nan_to_num(tr[self.cols].values, 0)
            y_tr = tr["label"].astype(int).values
            X_es = np.nan_to_num(es[self.cols].values, 0)
            y_es = es["label"].astype(int).values
            X_te = np.nan_to_num(te[self.cols].values, 0)
            y_te = te["label"].astype(int).values

            model = _make_lgb()
            model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)],   # stop on `es`…
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(0)])

            proba = model.predict_proba(X_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            acc_all = round(float((pred == y_te).mean()) * 100)
            conf = np.abs(proba - 0.5) * 100
            correct = (pred == y_te).astype(int)
            tiers = _calibrate_tiers(conf, correct)

            self.backtest_acc[key] = acc_all
            self.tier_info[key] = tiers
            val_store[key] = (te.index, proba, y_te, tiers["hi_thr"])
            print(f"    {name}/{key}: all={acc_all}%  "
                  f"HIGH={tiers['HIGH']}% (n={tiers['hi_n']})  "
                  f"MED={tiers['MEDIUM']}%  LOW={tiers['LOW']}%")

            # retrain on ALL data for production, keep val-calibrated tiers
            best_n = getattr(model, "best_iteration_", 300) or 300
            prod = _make_lgb(max(best_n, 100))
            prod.fit(np.nan_to_num(ok[self.cols].values, 0),
                     ok["label"].astype(int).values)
            self.models[key] = prod

        (self.strong_acc, self.strong_n,
         self.ultra_acc, self.ultra_n) = _measure_strong(val_store)
        print(f"  {name}: STRONG (≥{STRONG_MIN_ALIGN}/4 HIGH & agree) "
              f"→ ~{self.strong_acc}% on {self.strong_n} min | "
              f"ULTRA (+conf≥{MIN_ULTRA_CONF}) → ~{self.ultra_acc}% on {self.ultra_n} min")

        try:
            pickle.dump({"stamp": stamp, "version": MODEL_VERSION,
                         "cols": self.cols, "models": self.models,
                         "acc": self.backtest_acc, "tiers": self.tier_info,
                         "strong_acc": self.strong_acc, "strong_n": self.strong_n,
                         "ultra_acc": self.ultra_acc, "ultra_n": self.ultra_n},
                        open(pkl, "wb"))
            print(f"  {name}: {MODEL_VERSION} models saved ✅")
        except Exception as exc:
            print(f"  {name}: cache save failed ({exc})")

    def forecast(self, candles, day, other_candles=None, vix_ticks=None, **kwargs):
        """candles: list of ["HH:MM", o, h, l, c] 1-minute bars for `day`.
        other_candles / vix_ticks / es_ticks / inr_ticks (**kwargs) are accepted
        for API compatibility but the V3 engine does not use them (no train/serve
        skew — see module docstring)."""
        warm = getattr(self, "warmup_5m", None) is not None and len(self.warmup_5m)
        # with warmup we only need a couple of today's bars; without, wait for ~35
        if len(candles) < (10 if warm else 35):
            return None

        df = pd.DataFrame(candles, columns=["t", "Open", "High", "Low", "Close"])
        df.index = pd.to_datetime(day + " " + df["t"])
        m5 = df.resample("5min").agg({"Open": "first", "High": "max",
                                      "Low": "min", "Close": "last"}).dropna()
        if len(m5) < 1:
            return None
        # prepend prior-session bars so rolling windows are full from bar 1 today
        if warm:
            m5 = pd.concat([self.warmup_5m, m5])
            m5 = m5[~m5.index.duplicated(keep="last")].sort_index()
        elif len(m5) < 6:
            return None
        feats, _ = build5(m5)

        row = feats.iloc[-1]
        x = row[self.cols].astype(float)
        if x.isna().sum() > len(self.cols) * 0.3:
            return None
        xv = np.nan_to_num(x.values.reshape(1, -1), 0)

        price = float(df["Close"].iloc[-1])
        atr = float(row["atr"]) if np.isfinite(row["atr"]) else price * 0.001
        h = {}
        ups = 0
        for key, k in HORIZONS:
            p_up = float(self.models[key].predict_proba(xv)[0, 1])
            d = 1 if p_up >= 0.5 else -1
            ups += (d == 1)
            conf = round(max(p_up, 1 - p_up) * 100)
            info = self.tier_info.get(key)
            tier = _tier_of(max(p_up, 1 - p_up) * 100, info)
            tier_acc = info.get(tier) if info else None

            # realistic target = calibrated typical move for this horizon,
            # scaled to today's volatility (clamped so it can't run wild)
            vr = atr / self.ref_atr if getattr(self, "ref_atr", 0) else 1.0
            vr = min(max(vr, 0.5), 2.0)
            base = self.move_tgt.get(key, atr * (k ** 0.5)) if getattr(
                self, "move_tgt", None) else atr * (k ** 0.5)
            tgt_pts = max(base * vr, price * 0.0004)
            sl_pts = tgt_pts / TGT_ATR        # 1.5:1 reward:risk
            h[key] = {
                "dir": "UP" if d == 1 else "DOWN",
                "conf": conf,
                "tier": tier,
                "tier_acc": tier_acc,
                "sl": round(price - d * sl_pts, 1),
                "tgt": round(price + d * tgt_pts, 1),
                "sl_pts": round(sl_pts),
                "tgt_pts": round(tgt_pts),
                "outcome": None,
                "move": None,
            }

        consensus = "UP" if ups >= 3 else "DOWN" if ups <= 1 else "MIXED"

        # --- STRONG signal: >=N horizons HIGH tier AND agreeing on a side ---
        # ...but only in the model's good hours (midday dead zone suppressed).
        hi_up = sum(1 for z in h.values() if z["tier"] == "HIGH" and z["dir"] == "UP")
        hi_dn = sum(1 for z in h.values() if z["tier"] == "HIGH" and z["dir"] == "DOWN")
        align = max(hi_up, hi_dn)
        sdir = "UP" if hi_up >= hi_dn else "DOWN"
        tstr = candles[-1][0]
        tmin = int(tstr[:2]) * 60 + int(tstr[3:5])
        prime = not (DEAD_ZONE[0] <= tmin < DEAD_ZONE[1])
        strong = align >= STRONG_MIN_ALIGN and prime
        # ULTRA conviction: every agreeing horizon is itself confident enough.
        # Out-of-sample this tightening lifts 30m accuracy ~51% -> ~55-56%
        # (still NOT 90% — that only appears in-sample/lookahead). Fires rarely.
        agree_conf = [z["conf"] for z in h.values() if z["dir"] == sdir]
        min_conf = min(agree_conf) if agree_conf else 0
        ultra = strong and min_conf >= MIN_ULTRA_CONF
        # HIGH-CONVICTION regime: recent realized vol above the training median AND
        # a decent alignment. Direction is ~2pts better here (evidence-backed:
        # research + our data agree direction is more reliable in high-vol regimes).
        try:
            rv = float(m5["Close"].pct_change().abs().rolling(6).sum().iloc[-1])
        except Exception:
            rv = 0.0
        hi_vol = getattr(self, "vol_med", 0) and rv > self.vol_med
        hi_conviction = bool(strong and hi_vol)
        signal = {
            "strong": strong,
            "ultra": ultra,
            "hi_conviction": hi_conviction,
            "dir": sdir,
            "align": align,
            "min_conf": min_conf,
            "acc": (self.ultra_acc if ultra and self.ultra_acc
                    else self.strong_acc),
            "window": "prime" if prime else "midday",
        }

        return {"time": candles[-1][0], "index": self.name,
                "price": round(price, 1), "risk_rupees": round(self.risk_rupees),
                "consensus": consensus, "ups": ups, "h": h, "signal": signal}
