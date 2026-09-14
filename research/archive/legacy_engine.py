#!/usr/bin/env python3
"""
Legacy "Beast" forecast engine — the OLD approach (pre-July-26) rebuilt for a
fair head-to-head vs the current V3.1 engine.

It adds the old EXTRA features on top of BASE_COLS:
  - x_gap : cross-index divergence (this index vs the other, normalised)
  - vix, vix_pctl : India VIX level + rolling percentile
built the OLD way (from continuous long history at TRAIN, from today's intraday
+ VIX ticks at SERVE — i.e. WITH the train/serve skew intact, faithfully).

Same public API + output shape as forecast_engine.ForecastEngine, so both can be
scored identically. Used only by the dashboard's head-to-head logger.
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

import forecast_engine as fe   # reuse build5, helpers, constants
from oracle import paths

warnings.filterwarnings("ignore")
CACHE = paths.CACHE
LEGACY_COLS = fe.BASE_COLS + ["x_gap", "vix", "vix_pctl"]
MODEL_VERSION = "legacy1"


def _add_extras_train(feats, own_close, other_close, vix_close):
    other_close = other_close[~other_close.index.duplicated()].sort_index()
    oc = other_close.reindex(feats.index, method="ffill")
    feats = feats.copy()
    feats["x_gap"] = (own_close / own_close.iloc[0]) - (oc / oc.iloc[0])
    vix_close = vix_close[~vix_close.index.duplicated()].sort_index()
    vc = vix_close.reindex(feats.index, method="ffill")
    feats["vix"] = vc / 100.0
    feats["vix_pctl"] = vc.rolling(60).rank(pct=True)
    return feats


def _add_extras_serve(feats, own_close, other_candles, vix_ticks, day):
    feats = feats.copy()
    # cross-index gap from today's other-index candles (intraday only = the skew)
    try:
        odf = pd.DataFrame(other_candles, columns=["t", "o", "h", "l", "c"])
        odf.index = pd.to_datetime(day + " " + odf["t"])
        oc = odf["c"].resample("5min").last().dropna().reindex(feats.index, method="ffill")
        feats["x_gap"] = (own_close / own_close.iloc[0]) - (oc / oc.iloc[0])
    except Exception:
        feats["x_gap"] = 0.0
    try:
        vdf = pd.DataFrame(vix_ticks, columns=["t", "c"])
        vdf.index = pd.to_datetime(day + " " + vdf["t"])
        vc = vdf["c"].resample("5min").last().dropna().reindex(feats.index, method="ffill")
        feats["vix"] = vc / 100.0
        feats["vix_pctl"] = vc.rolling(60).rank(pct=True)
    except Exception:
        feats["vix"] = 0.0
        feats["vix_pctl"] = 0.5
    return feats


class LegacyForecastEngine:
    def __init__(self, name, capital, risk_pct, m5=None):
        self.name = name
        self.risk_rupees = capital * risk_pct / 100
        self.cols = LEGACY_COLS
        long = fe.load_long(name)
        m5 = long if long is not None else m5
        if m5 is None:
            raise RuntimeError("legacy engine needs long history")

        # warmup tail like the new engine
        try:
            w = m5.tail(60)[["Open", "High", "Low", "Close"]].copy()
            if getattr(w.index, "tz", None) is not None:
                w.index = w.index.tz_localize(None)
            self.warmup_5m = w
        except Exception:
            self.warmup_5m = None

        stamp = str(m5.index[-1].date())
        pkl = CACHE / f"{name}_legacy.pkl"
        if pkl.exists():
            try:
                s = pickle.load(open(pkl, "rb"))
                if s.get("stamp") == stamp and s.get("version") == MODEL_VERSION:
                    self.models = s["models"]; self.tier_info = s["tiers"]
                    print(f"  {name}: loaded LEGACY models")
                    return
            except Exception:
                pass

        print(f"  {name}: training LEGACY (extras) models…")
        feats, c5 = fe.build5(m5)
        other_long = fe.load_long(fe.OTHER[name]); vix_long = fe.load_long("INDIAVIX")
        feats = _add_extras_train(feats, c5, other_long["Close"], vix_long["Close"])

        self.models, self.tier_info = {}, {}
        days = sorted(set(feats.index.date))
        val_cut = days[-max(20, len(days) // 10)]
        for key, k in fe.HORIZONS:
            lab = (c5.shift(-k) > c5).astype(float)
            ok = feats.assign(label=lab).dropna(subset=self.cols + ["label"])
            tr = ok[ok.index.date < val_cut]; te = ok[ok.index.date >= val_cut]
            if len(tr) < 200 or len(te) < 50:
                m = fe._make_lgb(200)
                m.fit(np.nan_to_num(ok[self.cols].values, 0), ok["label"].astype(int).values)
                self.models[key] = m; self.tier_info[key] = None; continue
            Xtr = np.nan_to_num(tr[self.cols].values, 0); ytr = tr["label"].astype(int).values
            Xte = np.nan_to_num(te[self.cols].values, 0); yte = te["label"].astype(int).values
            m = fe._make_lgb()
            m.fit(Xtr, ytr, eval_set=[(Xte, yte)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            proba = m.predict_proba(Xte)[:, 1]; pred = (proba >= .5).astype(int)
            conf = np.abs(proba - .5) * 100
            self.tier_info[key] = fe._calibrate_tiers(conf, (pred == yte).astype(int))
            prod = fe._make_lgb(max(getattr(m, "best_iteration_", 300) or 300, 100))
            prod.fit(np.nan_to_num(ok[self.cols].values, 0), ok["label"].astype(int).values)
            self.models[key] = prod
        try:
            pickle.dump({"stamp": stamp, "version": MODEL_VERSION,
                         "models": self.models, "tiers": self.tier_info}, open(pkl, "wb"))
        except Exception:
            pass

    def forecast(self, candles, day, other_candles=None, vix_ticks=None, **kw):
        if len(candles) < 10:
            return None
        df = pd.DataFrame(candles, columns=["t", "Open", "High", "Low", "Close"])
        df.index = pd.to_datetime(day + " " + df["t"])
        m5 = df.resample("5min").agg({"Open": "first", "High": "max",
                                      "Low": "min", "Close": "last"}).dropna()
        if len(m5) < 1:
            return None
        if getattr(self, "warmup_5m", None) is not None and len(self.warmup_5m):
            m5 = pd.concat([self.warmup_5m, m5])
            m5 = m5[~m5.index.duplicated(keep="last")].sort_index()
        feats, c5 = fe.build5(m5)
        feats = _add_extras_serve(feats, c5, other_candles or [], vix_ticks or [], day)
        row = feats.iloc[-1]
        x = row[self.cols].astype(float)
        if x.isna().sum() > len(self.cols) * 0.4:
            return None
        xv = np.nan_to_num(x.values.reshape(1, -1), 0)
        price = float(df["Close"].iloc[-1]); ups = 0; h = {}
        for key, k in fe.HORIZONS:
            p_up = float(self.models[key].predict_proba(xv)[0, 1])
            d = 1 if p_up >= 0.5 else -1; ups += (d == 1)
            info = self.tier_info.get(key)
            tier = fe._tier_of(max(p_up, 1 - p_up) * 100, info)
            h[key] = {"dir": "UP" if d == 1 else "DOWN",
                      "conf": round(max(p_up, 1 - p_up) * 100), "tier": tier,
                      "outcome": None, "move": None}
        consensus = "UP" if ups >= 3 else "DOWN" if ups <= 1 else "MIXED"
        return {"time": candles[-1][0], "index": self.name, "price": round(price, 1),
                "consensus": consensus, "ups": ups, "h": h}
