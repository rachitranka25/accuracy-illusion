#!/usr/bin/env python3
"""Volatility prediction, held to the same standard as everything else.

The production volatility model reports 68% (BANKNIFTY) and 74% (NIFTY 50)
out-of-sample on a BIG-vs-QUIET target. Those numbers are real and reproduce
from the cached weights, but they were never put through the harness this
project applies to its directional work, and two things about them need
checking before they mean anything.

  1. The target is realised volatility over the next k bars, computed at EVERY
     bar. That is an overlapping label, exactly like the directional one, so the
     same variance inflation applies and the same per-event rescore is owed.

  2. Volatility clusters, so a persistence rule -- "the next 30 minutes look
     like the last 30" -- already scores well. An accuracy quoted without that
     baseline says almost nothing. This is the error the project documents
     elsewhere and it would be embarrassing to commit it here.

So the model is re-evaluated with: purged walk-forward, a shuffled-label
control, per-bar and per-event scoring, an effective sample size, and the
persistence baseline alongside.

Run:  python3 -m research.volatility_test [--horizon 30m]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from oracle import paths
from oracle.models import direction as fe
from research import common as C


def vol_dataset(name, horizon):
    """Features plus a BIG/QUIET label over the next k bars.

    Label is defined against the TRAINING median only, which is set inside the
    fold loop; here we return the raw realised volatility and let each fold
    threshold it on its own training data. Thresholding on the full sample would
    leak the test period's volatility level into the label.
    """
    k = C.HORIZON_BARS[horizon]
    bars = C.load_bars(name)
    feats, close = fe.build5(bars)
    X = feats[fe.BASE_COLS].copy()

    r = close.pct_change()
    fwd_vol = r.shift(-1).rolling(k).std().shift(-(k - 1))   # next k bars
    day = pd.Series(close.index.normalize(), index=close.index)
    fwd_vol[day.shift(-k).values != day.values] = np.nan     # no overnight

    # persistence baseline: volatility over the PAST k bars, known at time t
    past_vol = r.rolling(k).std()

    ok = X.notna().all(axis=1) & fwd_vol.notna() & past_vol.notna()
    return X[ok], fwd_vol[ok].values, past_vol[ok].values


def run(name, horizon, folds):
    k = C.HORIZON_BARS[horizon]
    rng = np.random.default_rng(C.SEED)
    X, fwd_vol, past_vol = vol_dataset(name, horizon)
    splits = list(C.walk_forward(len(X), k, folds))

    C.header(f"{name} — volatility (BIG vs QUIET), {horizon} horizon",
             f"{len(X):,} bars  |  {len(splits)} purged folds  |  "
             f"label thresholded on each fold's own training median")

    model = C.Score("LightGBM (per bar)")
    per_evt = C.Score("LightGBM (per event)")
    shuf = C.Score("LightGBM [shuffled]")
    persist = C.Score("persistence baseline")
    persist_evt = C.Score("persistence (per event)")
    n_effs = []

    for tr, te in splits:
        thresh = np.median(fwd_vol[tr])              # training median only
        y = (fwd_vol > thresh).astype(int)

        sc = StandardScaler().fit(X.iloc[tr])
        a, b = sc.transform(X.iloc[tr]), sc.transform(X.iloc[te])

        m = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               min_child_samples=40, random_state=C.SEED,
                               verbose=-1, n_jobs=1)
        m.fit(a, y[tr])
        p = m.predict(b)
        model.add(y[te], p)
        per_evt.add(y[te][::k], p[::k])
        h = (p == y[te]).astype(int)
        ne, _ = C.effective_n(h)
        n_effs.append(ne)

        ys = C.shuffled(y.copy(), rng)
        m2 = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                min_child_samples=40, random_state=C.SEED,
                                verbose=-1, n_jobs=1)
        m2.fit(a, ys[tr]); shuf.add(ys[te], m2.predict(b))

        # persistence: predict BIG iff the last k bars were above the same median
        pb = (past_vol[te] > thresh).astype(int)
        persist.add(y[te], pb)
        persist_evt.add(y[te][::k], pb[::k])

    n_eff = float(sum(n_effs))
    lo, hi = C.interval_at(model.acc, n_eff)
    plo, phi = C.interval_at(persist.acc, n_eff)

    print(f"{'estimator':<28}{'per-bar':>9}{'per-event':>11}{'n_eff':>9}{'CI at n_eff':>16}")
    print("-" * 73)
    print(f"{'LightGBM':<28}{model.acc*100:>8.1f}%{per_evt.acc*100:>10.1f}%"
          f"{n_eff:>9,.0f}{f'[{lo*100:.1f}, {hi*100:.1f}]':>16}")
    print(f"{'persistence baseline':<28}{persist.acc*100:>8.1f}%"
          f"{persist_evt.acc*100:>10.1f}%{n_eff:>9,.0f}"
          f"{f'[{plo*100:.1f}, {phi*100:.1f}]':>16}")
    print(f"{'shuffled-label control':<28}{shuf.acc*100:>8.1f}%{'—':>11}{'—':>9}{'—':>16}")

    edge_bar = (model.acc - persist.acc) * 100
    edge_evt = (per_evt.acc - persist_evt.acc) * 100
    print(f"\n  edge over persistence : {edge_bar:+.1f} pts per bar, "
          f"{edge_evt:+.1f} pts per event")
    print(f"  edge over shuffled    : {(model.acc - shuf.acc)*100:+.1f} pts")
    overlap = not (hi < plo or phi < lo)
    print(f"  model and baseline intervals at n_eff "
          f"{'OVERLAP — the edge is not established' if overlap else 'are disjoint'}")

    return {"index": name, "horizon": horizon, "bars": int(len(X)),
            "folds": len(splits),
            "model_per_bar": round(model.acc, 4),
            "model_per_event": round(per_evt.acc, 4),
            "persistence_per_bar": round(persist.acc, 4),
            "persistence_per_event": round(persist_evt.acc, 4),
            "shuffled": round(shuf.acc, 4),
            "n_eff": round(n_eff, 1),
            "model_ci_at_n_eff": [round(lo, 4), round(hi, 4)],
            "persistence_ci_at_n_eff": [round(plo, 4), round(phi, 4)],
            "edge_over_persistence_per_bar": round(edge_bar, 2),
            "edge_over_persistence_per_event": round(edge_evt, 2),
            "intervals_overlap": bool(overlap),
            "n_per_bar": model.n, "n_per_event": per_evt.n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="30m", choices=list(C.HORIZON_BARS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--indices", nargs="*", default=C.INDICES)
    a = ap.parse_args()

    out = [run(n, a.horizon, a.folds) for n in a.indices]

    print(f"\n{'='*78}\nVERDICT\n{'='*78}")
    for o in out:
        print(f"  {o['index']}: model {o['model_per_event']*100:.1f}% vs "
              f"persistence {o['persistence_per_event']*100:.1f}% per event "
              f"({o['edge_over_persistence_per_event']:+.1f} pts)")
    print(f"\n  The headline accuracy of a volatility model is mostly the baseline.")
    print(f"  What matters is the gap to persistence, and whether it survives")
    print(f"  being scored once per horizon rather than once per bar.")

    d = paths.ROOT / "results"; d.mkdir(exist_ok=True)
    f = d / f"volatility_test_{a.horizon}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {f.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
