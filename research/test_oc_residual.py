"""
The last honest question about option-chain data.

Earlier screening found dist_maxpain predicting next-day open->close at ~60%.
It also found dist_maxpain is 0.818 correlated with the plain 10-day return,
and that ret5 alone did the same job slightly better. So the apparent edge was
price mean-reversion wearing an options costume.

That test ran on 350 days, which cannot separate "no signal" from "signal too
small to see". This runs on roughly a decade and asks the sharper question:
strip out everything the price series already explains, and does the OI residual
predict anything on its own?

If the answer is no on ~2000 days, the option chain is closed as a source.
"""
import numpy as np
import pandas as pd
from scipy import stats
from oracle import paths

NAME = "BANKNIFTY"


def daily_chain(name=NAME):
    """Per-day chain summary from the long history."""
    d = pd.read_csv(paths.CACHE / f"{name}_oc_long.csv", parse_dates=["date"])
    d = d[(d["ce_oi"].fillna(0) > 0) | (d["pe_oi"].fillna(0) > 0)]

    out = []
    for day, g in d.groupby("date"):
        spot = g["spot"].dropna()
        if spot.empty:
            continue
        spot = spot.iloc[0]
        ce, pe = g["ce_oi"].fillna(0), g["pe_oi"].fillna(0)
        if ce.sum() <= 0 or pe.sum() <= 0:
            continue

        strikes = g["strike"].values
        # Max pain: strike minimising total intrinsic payout to option holders.
        pain = [(np.maximum(s - strikes, 0) * ce.values).sum()
                + (np.maximum(strikes - s, 0) * pe.values).sum() for s in strikes]
        mp = strikes[int(np.argmin(pain))]

        out.append({
            "date": day, "spot": spot,
            "pcr_oi": pe.sum() / ce.sum(),
            "pcr_vol": (g["pe_vol"].fillna(0).sum() + 1) / (g["ce_vol"].fillna(0).sum() + 1),
            "pcr_doi": ((g["pe_doi"].fillna(0).sum() - g["ce_doi"].fillna(0).sum())
                        / (g["pe_doi"].fillna(0).abs().sum()
                           + g["ce_doi"].fillna(0).abs().sum() + 1)),
            "dist_maxpain": spot / mp - 1,
            "dist_res": spot / g.loc[ce.idxmax(), "strike"] - 1,
            "dist_sup": spot / g.loc[pe.idxmax(), "strike"] - 1,
        })

    r = pd.DataFrame(out).sort_values("date").set_index("date")
    for c in ["pcr_oi", "pcr_vol"]:
        r[f"d_{c}"] = r[c].pct_change()
    return r.replace([np.inf, -np.inf], np.nan)


def price_block(chain, name=NAME):
    """Everything a price-only model already knows, on the same daily grid.

    Daily bars rather than resampled 5-min: the 5-min archive only reaches 2025
    and would throw away eight of the ten years of chain history.
    """
    d = pd.read_csv(paths.CACHE / f"{name}_daily_long.csv", index_col=0, parse_dates=True)
    d.index = d.index.tz_localize(None) if d.index.tz else d.index
    d = d.rename(columns=str.lower)
    d["oc"] = d["close"] / d["open"] - 1
    for k in [1, 2, 3, 5, 10, 20]:
        d[f"ret{k}"] = d["close"].pct_change(k)
    d["vol20"] = d["close"].pct_change().rolling(20).std()
    return d


def residualise(X, P):
    """Strip the linear price component out of each OI feature."""
    P = P.copy()
    P["const"] = 1.0
    R = pd.DataFrame(index=X.index)
    for c in X.columns:
        m = X[c].notna() & P.notna().all(axis=1)
        if m.sum() < 200:
            continue
        beta, *_ = np.linalg.lstsq(P[m].values, X.loc[m, c].values, rcond=None)
        r = pd.Series(np.nan, index=X.index)
        r[m] = X.loc[m, c].values - P[m].values @ beta
        R[c] = r
    return R


def screen(X, y, label, tag):
    print(f"\n{'='*76}\n{label}\n{'='*76}")
    print(f"{'feature':<16}{'IC':>8}{'p':>8}{'hit%':>8}{'H1':>8}{'H2':>8}  verdict")
    print("-" * 76)
    mid = len(y) // 2
    keep = []
    for c in X.columns:
        v, m = X[c], X[c].notna() & y.notna()
        if m.sum() < 300:
            continue
        ic, p = stats.spearmanr(v[m], y[m])
        lo, hi = v[m].quantile(0.2), v[m].quantile(0.8)
        ext = m & ((v <= lo) | (v >= hi))
        sig = np.sign(v[ext] - v[m].median()) * np.sign(ic if ic else 1)
        hit = (np.sign(y[ext]) == sig).mean() * 100 if ext.sum() else np.nan
        i1 = stats.spearmanr(v[m][:mid], y[m][:mid])[0]
        i2 = stats.spearmanr(v[m][mid:], y[m][mid:])[0]
        ok = (np.sign(i1) == np.sign(i2)) and min(abs(i1), abs(i2)) > 0.03 and p < 0.05
        if ok:
            keep.append(f"{c} [{tag}]")
        print(f"{c:<16}{ic:>8.3f}{p:>8.3f}{hit:>8.1f}{i1:>8.3f}{i2:>8.3f}  "
              f"{'** KEEP' if ok else ('~ unstable' if p < 0.05 else 'no')}")
    return keep


if __name__ == "__main__":
    chain = daily_chain()
    px = price_block(chain)
    df = chain.join(px, how="inner").dropna(subset=["oc"])
    print(f"{NAME}: {len(df)} days  {df.index.min().date()} -> {df.index.max().date()}")

    cols = ["pcr_oi", "pcr_vol", "pcr_doi", "dist_maxpain", "dist_res",
            "dist_sup", "d_pcr_oi", "d_pcr_vol"]
    X = df[[c for c in cols if c in df]]
    P = df[[f"ret{k}" for k in [1, 2, 3, 5, 10, 20]] + ["vol20"]]
    y = df["oc"].shift(-1)

    keep = screen(X, y, "RAW option-chain features -> next-day open->close", "raw")
    m = P.notna().all(axis=1)
    keep += screen(residualise(X[m], P[m]), y[m],
                   "RESIDUAL (price component removed) -> next-day open->close", "resid")

    print(f"\n{'='*76}\nBASELINE: what plain price returns do on the same days\n{'='*76}")
    screen(P[m], y[m], "price-only features", "price")

    print(f"\n{'='*76}\nVERDICT\n{'='*76}")
    print("  survivors:", keep if keep else "NONE")
