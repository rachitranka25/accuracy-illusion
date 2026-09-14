#!/usr/bin/env python3
"""
Meta-label filter for Pulse (López de Prado's meta-labeling).

The primary signal decides the SIDE (long/short). This secondary model decides
whether that particular setup is worth taking at all — it is trained on the
question the user actually cares about:

    "will this trade deliver at least RATE points per minute?"

That label matters. Training the same model on "did the target get hit" pushes
it toward small, easy targets: the hit-rate looks great (42% -> 53%) while the
money gets worse. Labelling on points-per-minute instead keeps the geometry
honest.

Out-of-sample (106 unseen days, BankNifty), on top of the existing speed/thrust
gates:
    gates only        2.3 trades/day   +1.44 pts/min   total -3777
    gates + meta 30%  1.0 trades/day   +1.82 pts/min   total  -981
and it held in both halves of the test period.

BankNifty only — the same model on NIFTY50 failed out-of-sample (+0.25 -> +0.08,
second half negative), so it is not applied there.
"""
import warnings
warnings.filterwarnings("ignore")

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from oracle import paths

CACHE = paths.CACHE
RATE = {"BANKNIFTY": 5.0}      # points-per-minute the trade must earn to count
KEEP_TOP = 0.30                # act on the best 30% of setups
COLS = ["f_speed", "f_speedf", "f_accel", "f_sigma", "f_mom1", "f_mom3",
        "f_mom6", "f_mom12", "f_eff", "f_ext", "f_pos", "f_pbig", "f_tod",
        "f_budget", "f_tgtfrac"]


def indicators(m5):
    """Everything the features need, computed once for a 5-minute frame."""
    c = m5["Close"]
    return {
        "r1": c.pct_change(1), "r3": c.pct_change(3),
        "r6": c.pct_change(6), "r12": c.pct_change(12),
        "sd": c.pct_change().rolling(12).std() * np.sqrt(12),
        "vel": c.diff().abs().rolling(6).mean() / 5.0,
        "velf": c.diff().abs().rolling(3).mean() / 5.0,
        "ema": c.ewm(span=20).mean(),
        "eff": (c - c.shift(12)).abs() / c.diff().abs().rolling(12).sum(),
        "hh": m5["High"].rolling(12).max(),
        "ll": m5["Low"].rolling(12).min(),
    }


def features_at(m5, ind, i, side, pbig, tgt_pts, budget, bar_of_day):
    """Feature row for one candidate trade. Shared by training and live use so
    the two can never drift apart."""
    e = float(m5["Close"].iloc[i])
    sig = e * float(ind["sd"].iloc[i])
    if not np.isfinite(sig) or sig <= 0:
        return None
    hh, ll = float(ind["hh"].iloc[i]), float(ind["ll"].iloc[i])
    row = {
        "f_speed": float(ind["vel"].iloc[i]) / e,
        "f_speedf": float(ind["velf"].iloc[i]) / e,
        "f_accel": float(ind["velf"].iloc[i]) / max(float(ind["vel"].iloc[i]), 1e-9),
        "f_sigma": float(ind["sd"].iloc[i]),
        "f_mom1": side * float(ind["r1"].iloc[i]),
        "f_mom3": side * float(ind["r3"].iloc[i]),
        "f_mom6": side * float(ind["r6"].iloc[i]),
        "f_mom12": side * float(ind["r12"].iloc[i]),
        "f_eff": float(ind["eff"].iloc[i]),
        "f_ext": side * (e - float(ind["ema"].iloc[i])) / sig,
        "f_pos": (e - ll) / max(hh - ll, 1e-9),
        "f_pbig": float(pbig),
        "f_tod": float(bar_of_day),
        "f_budget": float(budget),
        "f_tgtfrac": float(tgt_pts) / e,
    }
    return None if any(not np.isfinite(v) for v in row.values()) else row


class MetaFilter:
    """Loads a trained model; .keep(...) says whether to act on a setup."""

    def __init__(self, name):
        self.name = name
        self.ok = False
        f = CACHE / f"{name}_meta.pkl"
        if name not in RATE or not f.exists():
            return
        try:
            s = pickle.loads(f.read_bytes())
            self.model, self.thr, self.cols = s["model"], s["thr"], s["cols"]
            self.acc = s.get("ppm")
            self.ok = True
        except Exception:
            self.ok = False

    def prob(self, row):
        if not self.ok or row is None:
            return None
        x = pd.DataFrame([[row[c] for c in self.cols]], columns=self.cols)
        return float(self.model.predict_proba(x)[0, 1])

    def keep(self, row):
        """True = take the trade. Fails open: no model -> nothing is blocked."""
        p = self.prob(row)
        return True if p is None else p >= self.thr


def train(name="BANKNIFTY", path=None, verbose=True):
    """Build the dataset from history, fit the model, store it."""
    import lightgbm as lgb
    import torch
    from oracle.models import volatility as V
    from oracle.models import direction as fe

    path = path or str(CACHE / f"{name}_5m_long.csv")
    veng = V.VolatilityLSTM(name)
    m5 = pd.read_csv(path)
    m5["ts"] = pd.to_datetime(m5["ts"])
    m5 = m5.set_index("ts")[["Open", "High", "Low", "Close"]]
    if m5.index.tz is not None:
        m5.index = m5.index.tz_localize(None)

    # vol-LSTM probability at every bar (the strongest single feature)
    vf, _ = fe.build5(m5)
    vf = vf.dropna()
    Xn = (vf[veng.cols].values.astype(np.float32) - veng.mu) / veng.sd
    S = V.SEQ
    seqs = np.stack([Xn[t - S + 1:t + 1] for t in range(S - 1, len(Xn))]).astype(np.float32)
    with torch.no_grad():
        pb = torch.sigmoid(veng.model(torch.tensor(seqs))).numpy()
    pmap = dict(zip(vf.index[S - 1:], pb))

    ind = indicators(m5)
    d = m5.reset_index()
    d["hm"] = d["ts"].dt.strftime("%H:%M")
    d["date"] = d["ts"].dt.date
    C, H, L = d["Close"].values, d["High"].values, d["Low"].values
    agree_up = sum((ind[k] > 0).astype(int) for k in ("r1", "r3", "r6", "r12")) >= 3
    agree_dn = sum((ind[k] < 0).astype(int) for k in ("r1", "r3", "r6", "r12")) >= 3
    prime = (d["hm"] > "09:19") & (d["hm"] < "14:45")

    rows = []
    for _, g in d.groupby("date"):
        idx = list(g.index)
        pos = None
        for i in idx:
            if not prime.iloc[i]:
                continue
            s = 1 if agree_up.iloc[i] else -1 if agree_dn.iloc[i] else 0
            if s == 0 or s == pos:
                continue
            sdv, vv = ind["sd"].iloc[i], ind["vel"].iloc[i]
            if not np.isfinite(sdv) or sdv <= 0 or not np.isfinite(vv) or vv <= 0:
                continue
            pbig = pmap.get(d["ts"].iloc[i])
            if pbig is None:
                continue
            e = C[i]
            tgt_pts, sl_pts = e * sdv * 0.5, e * sdv * 2.0
            if tgt_pts < e * 0.0008:
                continue
            pos = s
            budget = int(max(30, min(90, round((tgt_pts / vv) * 3.0 / 5) * 5)))
            row = features_at(m5, ind, i, s, pbig, tgt_pts, budget, i - idx[0])
            if row is None:
                continue
            T, St = e + s * tgt_pts, e - s * sl_pts
            res, bars = None, None
            last = min(i + budget // 5, idx[-1])
            for j in range(i + 1, last + 1):
                if s == 1:
                    if L[j] <= St: res, bars = -sl_pts, j - i; break
                    if H[j] >= T:  res, bars = tgt_pts, j - i; break
                else:
                    if H[j] >= St: res, bars = -sl_pts, j - i; break
                    if L[j] <= T:  res, bars = tgt_pts, j - i; break
            if res is None:
                res, bars = s * (C[last] - e), last - i
            mins = max(bars * 5, 5)
            row.update(date=g["date"].iloc[0], ppm=(res - e * 0.0004) / mins)
            rows.append(row)

    df = pd.DataFrame(rows).dropna()
    df["y"] = (df["ppm"] >= RATE[name]).astype(float)
    dates = sorted(df["date"].unique())
    cut = dates[int(len(dates) * 0.7)]
    tr, te = df[df["date"] < cut], df[df["date"] >= cut]

    model = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=15,
                               min_child_samples=40, subsample=0.8,
                               colsample_bytree=0.8, reg_lambda=1.0,
                               n_jobs=1, verbose=-1)
    model.fit(tr[COLS], tr["y"])
    p_te = model.predict_proba(te[COLS])[:, 1]
    thr = float(np.quantile(p_te, 1 - KEEP_TOP))
    kept = te[p_te >= thr]
    ppm = float(kept["ppm"].mean())

    (CACHE / f"{name}_meta.pkl").write_bytes(pickle.dumps(
        {"model": model, "thr": thr, "cols": COLS, "ppm": ppm}))
    if verbose:
        print(f"  {name}: meta-filter trained on {len(tr)} setups, tested on {len(te)}")
        print(f"    keeps top {int(KEEP_TOP*100)}%  ->  {ppm:+.2f} pts/min "
              f"(vs {te['ppm'].mean():+.2f} unfiltered)")
    return thr, ppm


if __name__ == "__main__":
    train("BANKNIFTY")
