"""
Does intraday OI movement predict the next 10-30 minutes?

This is the one question the EOD data could not answer. Writers building calls
into a rally, or puts being unwound into a dip, is positioning information that
never reaches the price series until after the fact -- if it predicts anything,
it should show up here.

Twenty-eight days is thin, and the verdict says so. The stability rule still
applies: half the sample must agree with the other half.
"""
import numpy as np
import pandas as pd
from scipy import stats
from oracle import paths

NAME = "BANKNIFTY"
HORIZONS = [1, 2, 3]      # 10, 20, 30 minutes ahead


def load():
    oi = pd.read_csv(paths.CACHE / f"{NAME}_intraday_oi.csv", parse_dates=["ts"])
    oi = oi.set_index("ts").sort_index()

    b = pd.read_csv(paths.CACHE / f"{NAME}_5m.csv", index_col=0, parse_dates=True)
    b.index = b.index.tz_localize(None) if b.index.tz else b.index
    px = b["Close"].resample("10min").last().dropna()

    d = oi.join(px.rename("px"), how="inner")
    d["day"] = d.index.normalize()
    return d.dropna()


def features(d):
    g = d.groupby("day")
    f = pd.DataFrame(index=d.index)

    f["pcr"] = d["pe_oi"] / d["ce_oi"]
    # Level tells you little; the change in the level is the positioning event.
    f["d_pcr"] = g.apply(lambda x: (x["pe_oi"] / x["ce_oi"]).diff()).droplevel(0)
    f["d_pcr3"] = g.apply(lambda x: (x["pe_oi"] / x["ce_oi"]).diff(3)).droplevel(0)

    # Net fresh positioning: puts being added while calls are shed is bullish
    # commitment, and vice versa. Scaled so the number is comparable across days.
    dce = g["ce_oi"].diff()
    dpe = g["pe_oi"].diff()
    f["oi_flow"] = (dpe - dce) / (d["ce_oi"] + d["pe_oi"])
    f["oi_flow3"] = ((g["pe_oi"].diff(3) - g["ce_oi"].diff(3))
                     / (d["ce_oi"] + d["pe_oi"]))
    # Total OI expanding = new money engaging; contracting = unwind.
    f["oi_grow"] = (dce + dpe) / (d["ce_oi"] + d["pe_oi"])
    # Session-relative PCR: where today's PCR sits against today's own range.
    f["pcr_z"] = g.apply(
        lambda x: ((x["pe_oi"] / x["ce_oi"]) - (x["pe_oi"] / x["ce_oi"]).expanding().mean())
        / ((x["pe_oi"] / x["ce_oi"]).expanding().std() + 1e-9)).droplevel(0)
    return f.replace([np.inf, -np.inf], np.nan)


def screen(f, d, h):
    # Forward return, never crossing a day boundary.
    y = d.groupby("day")["px"].shift(-h) / d["px"] - 1
    print(f"\n{'='*72}\nforward {h*10} minutes   ({y.notna().sum()} obs)\n{'='*72}")
    print(f"{'feature':<12}{'IC':>8}{'p':>8}{'hit%':>8}{'H1':>8}{'H2':>8}  verdict")
    print("-" * 72)

    days = np.sort(d["day"].unique())
    mid = days[len(days) // 2]
    keep = []
    for c in f.columns:
        v = f[c]
        m = v.notna() & y.notna()
        if m.sum() < 100:
            continue
        ic, p = stats.spearmanr(v[m], y[m])
        lo, hi = v[m].quantile(0.25), v[m].quantile(0.75)
        ext = m & ((v <= lo) | (v >= hi))
        sig = np.sign(v[ext] - v[m].median()) * np.sign(ic if ic else 1)
        hit = (np.sign(y[ext]) == sig).mean() * 100 if ext.sum() else np.nan

        e1 = m & (d["day"] < mid)
        e2 = m & (d["day"] >= mid)
        i1 = stats.spearmanr(v[e1], y[e1])[0] if e1.sum() > 40 else np.nan
        i2 = stats.spearmanr(v[e2], y[e2])[0] if e2.sum() > 40 else np.nan
        stable = (np.sign(i1) == np.sign(i2)) and min(abs(i1), abs(i2)) > 0.04
        ok = stable and p < 0.10
        if ok:
            keep.append((c, h, ic, hit))
        print(f"{c:<12}{ic:>8.3f}{p:>8.3f}{hit:>8.1f}{i1:>8.3f}{i2:>8.3f}  "
              f"{'** KEEP' if ok else ('~ unstable' if p < 0.10 else 'no')}")
    return keep


if __name__ == "__main__":
    d = load()
    f = features(d)
    print(f"{NAME}: {len(d)} points, {d['day'].nunique()} days "
          f"({d.index.min().date()} -> {d.index.max().date()})")

    survivors = []
    for h in HORIZONS:
        survivors += screen(f, d, h)

    print(f"\n{'='*72}\nSURVIVORS\n{'='*72}")
    if survivors:
        for c, h, ic, hit in survivors:
            print(f"  {c} @ {h*10}min   IC={ic:+.3f}  hit={hit:.1f}%")
    else:
        print("  NONE -- no intraday OI feature held across both halves")
