"""
Screen the positioning features one at a time, honestly.

The rule that has killed every idea so far stays: a feature is only real if it
works in the FIRST half and the SECOND half. One good number over the whole
sample is what overfitting looks like from the inside.

Features are known at the close of day T. Targets live on day T+1. The shift is
the only thing standing between this and a lie, so it happens once, here.
"""
import numpy as np
import pandas as pd
from scipy import stats

from oracle.features import positioning
from oracle import paths

NAME = "BANKNIFTY"


def load_targets(name):
    """Next-day moves, built from the same 5-min bars the live system uses."""
    b = pd.read_csv(paths.CACHE / f"{name}_5m_long.csv", index_col=0, parse_dates=True)
    b.index = b.index.tz_localize(None) if b.index.tz else b.index
    b = b[(b.index.time >= pd.Timestamp("09:15").time()) &
          (b.index.time <= pd.Timestamp("15:30").time())]
    g = b.groupby(b.index.normalize())
    d = pd.DataFrame({
        "open": g["Open"].first(), "close": g["Close"].last(),
        "high": g["High"].max(), "low": g["Low"].min(),
    })
    # open->close is what an intraday system can actually capture; the
    # overnight gap is not tradeable from a 09:20 entry.
    d["oc"] = d["close"] / d["open"] - 1
    d["cc"] = d["close"].pct_change()
    return d


def screen(X, y, label):
    print(f"\n{'='*74}\n{label}   ({len(y)} days)\n{'='*74}")
    print(f"{'feature':<16}{'IC':>8}{'p':>8}{'hit%':>8}{'H1 IC':>9}{'H2 IC':>9}  verdict")
    print("-" * 74)

    mid = len(y) // 2
    keep = []
    for c in X.columns:
        v = X[c]
        m = v.notna() & y.notna()
        if m.sum() < 120:
            continue
        ic, p = stats.spearmanr(v[m], y[m])

        # Directional hit rate of the sign-following rule, using only the
        # extreme quintiles where a signal would actually be acted on.
        lo, hi = v[m].quantile(0.2), v[m].quantile(0.8)
        ext = m & ((v <= lo) | (v >= hi))
        sig = np.sign(v[ext] - v[m].median()) * np.sign(ic if ic else 1)
        hit = (np.sign(y[ext]) == sig).mean() * 100 if ext.sum() else np.nan

        i1 = stats.spearmanr(v[m][:mid], y[m][:mid])[0]
        i2 = stats.spearmanr(v[m][mid:], y[m][mid:])[0]
        stable = np.sign(i1) == np.sign(i2) and min(abs(i1), abs(i2)) > 0.04
        ok = stable and p < 0.10
        if ok:
            keep.append(c)
        print(f"{c:<16}{ic:>8.3f}{p:>8.3f}{hit:>8.1f}{i1:>9.3f}{i2:>9.3f}  "
              f"{'** KEEP' if ok else ('~ unstable' if p < 0.10 else 'no')}")
    return keep


if __name__ == "__main__":
    pos = positioning.build(NAME)
    tgt = load_targets(NAME)

    df = pos.join(tgt, how="inner")
    cols = [c for c in positioning.FEATURES if c in df]
    X = df[cols]

    survivors = {}
    for tname in ["oc", "cc"]:
        # THE shift: features from day T, outcome from day T+1.
        y = df[tname].shift(-1)
        survivors[tname] = screen(X, y, f"predicting NEXT DAY {tname}")

    print(f"\n{'='*74}\nSURVIVORS (stable across both halves)\n{'='*74}")
    for k, v in survivors.items():
        print(f"  {k}: {v if v else 'NONE'}")
