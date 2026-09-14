#!/usr/bin/env python3
"""
Volatility engine — predicts the MAGNITUDE regime, not direction.

For each horizon it answers one binary question:
    "Will realized volatility over the next k bars be ABOVE the (training)
     median, or below?"   → HIGH-VOL  /  LOW-VOL

Why: direction on 5-60m index bars is ~50% (a coin toss, confirmed repeatedly).
Volatility, by contrast, CLUSTERS (calm follows calm, storms follow storms) —
the single strongest stylised fact in finance. On this project's own data a
90-day walk-forward holds at ~62-82% (vs direction's 50%), and it did NOT
collapse the way meta-labeling did.

HONEST LIMITS:
  - A high raw hit-rate is partly regime persistence: the naive "same as
    recent" baseline is already ~63-80% because vol clusters. The MODEL's real
    contribution is the edge over that baseline — biggest on NIFTY50 30m/60m
    (~+9 to +12 pts), small on BankNifty. `edge` is stored per horizon.
  - Predicting volatility is NOT a trade by itself — you cannot buy "high vol".
    It is a COMPONENT: feed it to an options straddle (buy vol when HIGH) or use
    it as an ORB regime filter (breakout on HIGH-vol days, fade on LOW).

Reuses build5 + BASE_COLS so it is apples-to-apples with the direction engines.
Same output-ish shape + warmup as forecast_engine, but a separate scoreboard.
"""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from oracle.models import direction as fe
from oracle import paths

warnings.filterwarnings("ignore")
CACHE = paths.CACHE
MODEL_VERSION = "vol1"


class VolatilityEngine:
    def __init__(self, name, m5=None):
        self.name = name
        self.cols = fe.BASE_COLS
        long = fe.load_long(name)
        m5 = long if long is not None else m5
        if m5 is None:
            raise RuntimeError("volatility engine needs long history")

        try:
            w = m5.tail(60)[["Open", "High", "Low", "Close"]].copy()
            if getattr(w.index, "tz", None) is not None:
                w.index = w.index.tz_localize(None)
            self.warmup_5m = w
        except Exception:
            self.warmup_5m = None

        stamp = str(m5.index[-1].date())
        pkl = CACHE / f"{name}_vol.pkl"
        if pkl.exists():
            try:
                s = pickle.load(open(pkl, "rb"))
                if s.get("stamp") == stamp and s.get("version") == MODEL_VERSION:
                    self.models = s["models"]; self.thr = s["thr"]
                    self.acc = s["acc"]; self.edge = s["edge"]
                    print(f"  {name}: loaded VOL models (acc {self.acc})")
                    return
            except Exception:
                pass

        print(f"  {name}: training VOLATILITY models…")
        feats, c5 = fe.build5(m5)
        ret = c5.pct_change(); rv = ret.pow(2)
        days = sorted(set(feats.index.date))
        val_cut = days[-max(20, len(days) // 10)]

        self.models, self.thr, self.acc, self.edge = {}, {}, {}, {}
        for key, k in fe.HORIZONS:
            fv = np.sqrt(rv.rolling(k).sum().shift(-k))     # realized vol over next k
            df = feats.assign(fv=fv).dropna(subset=self.cols + ["fv"])
            tr = df[df.index.date < val_cut]; te = df[df.index.date >= val_cut]
            thr = float(tr["fv"].median())                  # threshold from TRAIN only
            self.thr[key] = thr
            ytr = (tr["fv"] > thr).astype(int).values
            yte = (te["fv"] > thr).astype(int).values
            Xtr = np.nan_to_num(tr[self.cols].values, 0)
            Xte = np.nan_to_num(te[self.cols].values, 0)
            m = fe._make_lgb()
            m.fit(Xtr, ytr, eval_set=[(Xte, yte)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(0)])
            p = m.predict_proba(Xte)[:, 1]
            acc = round(float(((p >= .5).astype(int) == yte).mean()) * 100)
            base = round(max(yte.mean(), 1 - yte.mean()) * 100)
            self.acc[key] = acc; self.edge[key] = acc - base
            print(f"    {name}/{key}: vol-acc={acc}%  (baseline {base}%, edge {acc-base:+d})")
            # retrain on all data for production
            prod = fe._make_lgb(max(getattr(m, "best_iteration_", 300) or 300, 100))
            allok = df
            prod.fit(np.nan_to_num(allok[self.cols].values, 0),
                     (allok["fv"] > thr).astype(int).values)
            self.models[key] = prod
        try:
            pickle.dump({"stamp": stamp, "version": MODEL_VERSION, "models": self.models,
                         "thr": self.thr, "acc": self.acc, "edge": self.edge}, open(pkl, "wb"))
        except Exception:
            pass

    def predict(self, candles, day, **kw):
        """candles: list of ["HH:MM", o,h,l,c]. Returns {horizon: {vol, prob, acc, edge}}."""
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
        feats, _ = fe.build5(m5)
        row = feats.iloc[-1]
        x = row[self.cols].astype(float)
        if x.isna().sum() > len(self.cols) * 0.4:
            return None
        xv = np.nan_to_num(x.values.reshape(1, -1), 0)
        out = {}
        for key, k in fe.HORIZONS:
            p = float(self.models[key].predict_proba(xv)[0, 1])
            out[key] = {"vol": "HIGH" if p >= 0.5 else "LOW",
                        "prob": round(p * 100),
                        "acc": self.acc.get(key), "edge": self.edge.get(key)}
        return {"time": candles[-1][0], "index": self.name, "h": out}


if __name__ == "__main__":
    for n in ["BANKNIFTY", "NIFTY50"]:
        VolatilityEngine(n)
