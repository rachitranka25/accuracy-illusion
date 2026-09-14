"""
Out-of-sample test of the pinning signal.

The screen said distance-from-OI-cluster mean-reverts intraday. dist_maxpain,
dist_res and dist_sup are three views of one idea and move together, so they get
combined into a single z-score rather than counted as three discoveries.

Nothing is fitted here. The rule is fixed in advance -- fade the stretch -- and
the only thing measured is what it would have paid on days the model never saw.
Train window sets the z-score scale, that is all.
"""
import numpy as np
import pandas as pd

from oracle.features import positioning
from research.screen_positioning import load_targets

NAME = "BANKNIFTY"
SPLIT = 0.60      # first 60% of days sets scale, last 40% is untouched
ENTRY = 1.0       # act only when the stretch is at least this many sigma


def build_signal(df, train_end):
    """Fade the stretch: spot far above the OI cluster -> short, far below -> long."""
    cl = ["dist_maxpain", "dist_res", "dist_sup"]
    tr = df.iloc[:train_end]
    z = pd.DataFrame(index=df.index)
    for c in cl:
        mu, sd = tr[c].mean(), tr[c].std()
        z[c] = (df[c] - mu) / sd if sd else 0.0
    stretch = z.mean(axis=1)
    return -stretch          # negative IC -> fade


def run(sig, y, mask, label):
    m = mask & sig.notna() & y.notna()
    if m.sum() < 20:
        print(f"{label}: too few days"); return
    s, r = sig[m], y[m]
    take = s.abs() >= ENTRY
    if take.sum() < 10:
        print(f"{label}: too few triggers"); return
    d = np.sign(s[take])
    pnl = d * r[take] * 100                      # percent points per trade
    hit = (pnl > 0).mean() * 100
    print(f"{label:<26} n={take.sum():>4}  hit={hit:>5.1f}%  "
          f"avg={pnl.mean():>+6.3f}%  total={pnl.sum():>+7.2f}%  "
          f"sharpe={pnl.mean()/pnl.std()*np.sqrt(252):>5.2f}")
    return pnl


if __name__ == "__main__":
    pos = positioning.build(NAME)
    df = pos.join(load_targets(NAME), how="inner").dropna(subset=["dist_maxpain"])
    y = df["oc"].shift(-1)                       # act tomorrow on today's close info

    n = len(df)
    tr_end = int(n * SPLIT)
    sig = build_signal(df, tr_end)
    is_m = pd.Series(False, index=df.index); is_m.iloc[:tr_end] = True
    oos_m = ~is_m

    print(f"{NAME}: {n} days  |  in-sample {tr_end}  out-of-sample {n-tr_end}")
    print(f"OOS period: {df.index[tr_end].date()} -> {df.index[-1].date()}\n")

    run(sig, y, is_m,  "IN-SAMPLE")
    oos = run(sig, y, oos_m, "OUT-OF-SAMPLE")

    # Baseline: what buy-and-hold-the-open did on the same OOS days.
    b = y[oos_m].dropna() * 100
    print(f"{'(baseline: always long)':<26} n={len(b):>4}  "
          f"hit={(b>0).mean()*100:>5.1f}%  avg={b.mean():>+6.3f}%")

    if oos is not None:
        print(f"\nOOS by year:")
        for yr, g in oos.groupby(oos.index.year):
            print(f"   {yr}  n={len(g):>3}  hit={(g>0).mean()*100:>5.1f}%  "
                  f"total={g.sum():>+7.2f}%")
