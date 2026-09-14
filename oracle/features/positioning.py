"""
Positioning features from options open interest and participant data.

Everything the forecaster currently sees is a transform of the same 5-min price
series -- rsi, ema, macd, bollinger, atr, vwap. Fifteen columns, one source of
information. That is the reason it sits at a coin flip: no amount of reshaping
price tells you anything price did not already say.

Open interest is different. It is what people have actually committed money to,
and it is free in the EOD bhavcopy we already keep on disk. Same for the
participant file (FII / DII / Client / Pro net futures).

Every feature here is built from data available at the CLOSE of day T and is
only ever used to predict day T+1. Nothing peeks.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from oracle import paths

CACHE = paths.CACHE
def _nearest_expiry(g):
    """Front-month chain only -- back months carry stale OI that drowns the signal."""
    return g[g["XpryDt"] == g["XpryDt"].min()]


def option_features(name="BANKNIFTY"):
    f = CACHE / f"{name}_options_eod.csv"
    if not f.exists():
        return pd.DataFrame()

    df = pd.read_csv(f, parse_dates=["TradDt", "XpryDt"])
    df = df[df["OptnTp"].isin(["CE", "PE"])]
    df = df[df["OpnIntrst"] > 0]

    out = []
    for day, g in df.groupby("TradDt"):
        g = _nearest_expiry(g)
        if len(g) < 20:
            continue
        spot = g["UndrlygPric"].iloc[0]
        if not spot or spot <= 0:
            continue

        ce = g[g["OptnTp"] == "CE"]
        pe = g[g["OptnTp"] == "PE"]
        if ce.empty or pe.empty:
            continue

        ce_oi, pe_oi = ce["OpnIntrst"].sum(), pe["OpnIntrst"].sum()
        ce_vol, pe_vol = ce["TtlTradgVol"].sum(), pe["TtlTradgVol"].sum()
        ce_doi, pe_doi = ce["ChngInOpnIntrst"].sum(), pe["ChngInOpnIntrst"].sum()

        # Where the biggest walls sit. This is support/resistance drawn by money
        # rather than by hand: peak call OI caps rallies, peak put OI floors dips.
        res = ce.loc[ce["OpnIntrst"].idxmax(), "StrkPric"]
        sup = pe.loc[pe["OpnIntrst"].idxmax(), "StrkPric"]

        # Max pain: the strike where total option-holder payout is smallest.
        strikes = np.sort(g["StrkPric"].unique())
        ce_by = ce.groupby("StrkPric")["OpnIntrst"].sum()
        pe_by = pe.groupby("StrkPric")["OpnIntrst"].sum()
        pain = []
        for s in strikes:
            c = (np.maximum(s - ce_by.index.values, 0) * ce_by.values).sum()
            p = (np.maximum(pe_by.index.values - s, 0) * pe_by.values).sum()
            pain.append(c + p)
        max_pain = strikes[int(np.argmin(pain))]

        # ATM straddle price as a share of spot -- the market's own quote for
        # how far it expects to travel. A cheap free proxy for implied vol.
        atm = strikes[np.argmin(np.abs(strikes - spot))]
        ac = ce[ce["StrkPric"] == atm]["ClsPric"]
        ap = pe[pe["StrkPric"] == atm]["ClsPric"]
        strad = (ac.iloc[0] + ap.iloc[0]) / spot if len(ac) and len(ap) else np.nan

        out.append({
            "date": day.normalize(),
            "spot": spot,
            "pcr_oi": pe_oi / ce_oi if ce_oi else np.nan,
            "pcr_vol": pe_vol / ce_vol if ce_vol else np.nan,
            # Fresh money is more informative than the accumulated book.
            "pcr_doi": (pe_doi - ce_doi) / (abs(pe_doi) + abs(ce_doi) + 1),
            "dist_maxpain": spot / max_pain - 1,
            "dist_res": spot / res - 1,
            "dist_sup": spot / sup - 1,
            "straddle": strad,
            "oi_total": ce_oi + pe_oi,
        })

    d = pd.DataFrame(out).sort_values("date").reset_index(drop=True)
    if d.empty:
        return d
    # Day-over-day moves matter more than levels: a PCR of 0.9 means nothing on
    # its own, a PCR that jumped from 0.6 to 0.9 overnight means something.
    for c in ["pcr_oi", "pcr_vol", "straddle", "oi_total"]:
        d[f"d_{c}"] = d[c].pct_change()
    return d


def participant_features():
    f = CACHE / "participant_oi.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f, parse_dates=["date"]).sort_values("date")

    def net(who):
        lo, sh = d[f"{who}_fut_long"], d[f"{who}_fut_short"]
        return (lo - sh) / (lo + sh + 1)

    out = pd.DataFrame({"date": d["date"].dt.normalize()})
    for who in ["FII", "DII", "Client", "Pro"]:
        out[f"net_{who}"] = net(who).values
        out[f"dnet_{who}"] = out[f"net_{who}"].diff()
    # Retail long while foreign money is short is the classic squeeze setup.
    out["fii_vs_client"] = out["net_FII"] - out["net_Client"]
    return out.reset_index(drop=True)


def futures_features(name="BANKNIFTY"):
    f = CACHE / f"{name}_futures_eod.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f, parse_dates=["TradDt", "XpryDt"])
    d = d.sort_values(["TradDt", "XpryDt"]).groupby("TradDt").first().reset_index()
    out = pd.DataFrame({"date": d["TradDt"].dt.normalize()})
    # Premium of futures over spot -- positive is carry/bullish leaning.
    out["basis"] = (d["ClsPric"] / d["UndrlygPric"] - 1).values
    out["d_basis"] = out["basis"].diff()
    out["fut_doi"] = (d["ChngInOpnIntrst"] / d["OpnIntrst"].replace(0, np.nan)).values
    return out


def build(name="BANKNIFTY"):
    """Daily positioning table. Index = date, columns = features known at that close."""
    o, p, fu = option_features(name), participant_features(), futures_features(name)
    d = o
    for other in (p, fu):
        if not other.empty:
            d = d.merge(other, on="date", how="left")
    return d.set_index("date").replace([np.inf, -np.inf], np.nan)


FEATURES = [
    "pcr_oi", "pcr_vol", "pcr_doi", "dist_maxpain", "dist_res", "dist_sup",
    "straddle", "d_pcr_oi", "d_pcr_vol", "d_straddle", "d_oi_total",
    "net_FII", "dnet_FII", "net_Client", "dnet_Client", "fii_vs_client",
    "basis", "d_basis", "fut_doi",
]


if __name__ == "__main__":
    d = build()
    print(d[["spot"] + [c for c in FEATURES if c in d]].tail(10).to_string())
    print(f"\n{len(d)} days, {d.index.min().date()} -> {d.index.max().date()}")
