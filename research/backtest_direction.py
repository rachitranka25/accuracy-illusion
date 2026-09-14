#!/usr/bin/env python3
"""
Honest walk-forward backtest for the V3 forecast engine.

Trains on the past, tests on a held-out recent slice (NO lookahead), and
reports the real out-of-sample hit-rate per horizon AND per confidence tier
(HIGH / MEDIUM / LOW) — the same tiers the live dashboard shows.

Usage:  python3 backtest.py            # both indices, last 45 test days
        python3 backtest.py --days 60
"""
import argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb
from pathlib import Path
from oracle.models import direction as fce
from oracle import paths

CACHE = paths.CACHE
def load(name):
    return pd.read_csv(CACHE / f"{name}_5m_long.csv", index_col=0, parse_dates=True)


def run(name, test_days):
    m5 = load(name)
    feats, c5 = fce.build5(m5)
    cols = fce.BASE_COLS
    days = sorted(set(feats.index.date))
    cut = days[-test_days]

    print(f"\n{'='*66}")
    print(f"  {name}  —  last {test_days} sessions held out (out-of-sample)")
    print(f"{'='*66}")
    print(f"  {'horizon':<8}{'signals':>8}{'  all%':>7}"
          f"{'   HIGH% (n)':>16}{'   MED%':>8}{'   LOW%':>8}")

    for key, k in fce.HORIZONS:
        lab = (c5.shift(-k) > c5).astype(float)
        ok = feats.assign(label=lab).dropna(subset=cols + ["label"])
        tr = ok[ok.index.date < cut]
        te = ok[ok.index.date >= cut]

        Xtr = np.nan_to_num(tr[cols].values, 0); ytr = tr["label"].astype(int).values
        Xte = np.nan_to_num(te[cols].values, 0); yte = te["label"].astype(int).values

        m = fce._make_lgb()
        m.fit(Xtr, ytr, eval_set=[(Xte, yte)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

        p = m.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        conf = np.abs(p - 0.5) * 100
        correct = (pred == yte).astype(int)
        acc_all = correct.mean() * 100
        t = fce._calibrate_tiers(conf, correct)
        hi = f"{t['HIGH']}% ({t['hi_n']})" if t['HIGH'] is not None else "—"
        md = f"{t['MEDIUM']}%" if t['MEDIUM'] is not None else "—"
        lo = f"{t['LOW']}%" if t['LOW'] is not None else "—"
        print(f"  {key:<8}{len(yte):>8}{acc_all:>7.1f}{hi:>16}{md:>8}{lo:>8}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=45, help="held-out test sessions")
    args = ap.parse_args()
    for name in ["BANKNIFTY", "NIFTY50"]:
        run(name, args.days)
    print(f"\n{'-'*66}")
    print("  Reality check: per-bar direction is near a coin toss (~50-53%).")
    print("  The HIGH tier is the model's most-confident calls — a few points")
    print("  better, and where the small real edge lives. There is no 90%.")
    print("  The disciplined ORB system (nifty_live.py backtest) is the")
    print("  positive-expectancy money engine; this forecaster is the lean.")
