#!/usr/bin/env python3
"""Short-premium selling, held to the standard the rest of this paper uses.

An earlier version of this work reported the variance risk premium as a clean
positive with a mean and a win rate and nothing else: no dispersion, no interval,
no correction for the fact that several entry rules were tried, and no check on
whether the two indices were even comparable. Those are the same omissions the
paper criticises elsewhere, so they are closed here.

What this adds over a bare mean:

  DISPERSION      Standard deviation, t-statistic and a one-sided p-value for
                  profitability, plus a bootstrap interval on the mean that
                  makes no normality assumption. Short-premium P&L is sharply
                  left-skewed, so a t-test alone is not enough.

  MULTIPLICITY    The entry rule is a choice, and choices were tried. Every
                  configuration in the grid is reported, Benjamini-Hochberg is
                  applied across them, and a block-bootstrap null answers the
                  real question: what would the BEST of this many configurations
                  look like if premium selling had no edge at all?

  REGIME          NSE moved BANKNIFTY options from weekly to monthly expiry
                  during the sample. Comparing a weekly series to a monthly one
                  is not a like-for-like test, so expiry spacing is measured and
                  the sample is split on it.

Run:  python3 -m research.vrp_test
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from scipy import stats

from oracle import paths
from research import common as C

COST_FRACTIONS = (0.05, 0.10, 0.20)      # round-trip, as a share of premium sold
ENTRY_DTE = (1, 2, 3, 5)                 # trading days before expiry


def straddles(name, dte):
    """One short ATM straddle per expiry, entered `dte` trading days before it."""
    f = paths.CACHE / f"{name}_options_eod.csv"
    if not f.exists():
        return pd.DataFrame()
    o = pd.read_csv(f, usecols=["TradDt", "XpryDt", "StrkPric", "OptnTp",
                                "ClsPric", "UndrlygPric"])
    o["XpryDt"] = pd.to_datetime(o["XpryDt"]); o["TradDt"] = pd.to_datetime(o["TradDt"])

    rows = []
    for exp, grp in o.groupby("XpryDt"):
        days = sorted(grp["TradDt"].unique())
        if len(days) <= dte:
            continue
        ref = days[-1 - dte]
        day = grp[grp["TradDt"] == ref]
        if day.empty:
            continue
        spot = day["UndrlygPric"].median()
        if not np.isfinite(spot) or spot <= 0:
            continue
        strikes = day["StrkPric"].unique()
        atm = strikes[np.argmin(np.abs(strikes - spot))]
        ce = day[(day.StrkPric == atm) & (day.OptnTp == "CE")]["ClsPric"]
        pe = day[(day.StrkPric == atm) & (day.OptnTp == "PE")]["ClsPric"]
        if ce.empty or pe.empty:
            continue
        premium = float(ce.iloc[0]) + float(pe.iloc[0])
        if premium <= 0:
            continue
        exp_rows = grp[grp["TradDt"] == exp]
        if exp_rows.empty:
            continue
        settle = exp_rows["UndrlygPric"].median()
        if not np.isfinite(settle):
            continue
        rows.append({"expiry": exp, "entry": ref, "premium": premium,
                     "move": abs(settle - atm), "gross": premium - abs(settle - atm)})

    df = pd.DataFrame(rows).sort_values("expiry").reset_index(drop=True)
    if len(df):
        df["gap_days"] = df["expiry"].diff().dt.days
    return df


def evaluate(df, cost_frac):
    """Net P&L per trade with dispersion, not just a mean."""
    net = (df["gross"] - df["premium"] * cost_frac).values
    n = len(net)
    if n < 15:
        return None
    mean, sd = float(net.mean()), float(net.std(ddof=1))
    t = mean / (sd / np.sqrt(n)) if sd > 0 else 0.0
    p_one = 1 - stats.t.cdf(t, df=n - 1)

    rng = np.random.default_rng(C.SEED)
    boot = np.array([net[rng.integers(0, n, n)].mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    eq = np.cumsum(net)
    dd = float((eq - np.maximum.accumulate(eq)).min())

    return {"n": n, "win_rate": float((net > 0).mean()),
            "mean": mean, "sd": sd, "t": float(t), "p_one_sided": float(p_one),
            "ci95": [float(lo), float(hi)],
            "sharpe_per_trade": float(mean / sd) if sd > 0 else 0.0,
            "worst": float(net.min()), "max_drawdown": dd,
            "skew": float(stats.skew(net)),
            "h1_mean": float(net[:n // 2].mean()),
            "h2_mean": float(net[n // 2:].mean())}


def bootstrap_best_null(all_net, n_boot=2000, rng=None):
    """What would the best of these configurations look like with no edge?

    Each configuration's P&L is recentred to zero mean and resampled. The
    maximum mean across configurations is recorded, giving the distribution of
    "best of N" under a true null, in the same spirit as the strategy sweep.
    """
    rng = rng or np.random.default_rng(C.SEED)
    centred = [v - v.mean() for v in all_net]
    best = np.empty(n_boot)
    for b in range(n_boot):
        best[b] = max(v[rng.integers(0, len(v), len(v))].mean() for v in centred)
    return best


def run(name, cost_frac):
    print(f"\n{'='*84}")
    print(f"{name} — short ATM straddle, cost {cost_frac*100:.0f}% of premium")
    print(f"{'='*84}")
    print(f"  {'entry':<8}{'n':>5}{'win':>7}{'net/trade':>11}{'sd':>8}{'t':>7}"
          f"{'p':>8}{'95% CI':>18}{'worst':>9}{'h1/h2':>14}")
    print("  " + "-" * 82)

    rows, nets = [], []
    for dte in ENTRY_DTE:
        df = straddles(name, dte)
        if df.empty:
            continue
        r = evaluate(df, cost_frac)
        if r is None:
            continue
        r["dte"] = dte
        r["median_gap_days"] = float(df["gap_days"].median()) if len(df) > 1 else float("nan")
        rows.append(r)
        nets.append((df["gross"] - df["premium"] * cost_frac).values)
        lo, hi = r["ci95"]
        print(f"  {dte:<8}{r['n']:>5}{r['win_rate']*100:>6.0f}%{r['mean']:>11.1f}"
              f"{r['sd']:>8.0f}{r['t']:>7.2f}{r['p_one_sided']:>8.3f}"
              f"{f'[{lo:.0f}, {hi:.0f}]':>18}{r['worst']:>9.0f}"
              f"{f'{r[chr(104)+chr(49)+chr(95)+chr(109)+chr(101)+chr(97)+chr(110)]:.0f}/{r[chr(104)+chr(50)+chr(95)+chr(109)+chr(101)+chr(97)+chr(110)]:.0f}':>14}")

    if not rows:
        return None

    ps = np.array([r["p_one_sided"] for r in rows])
    order = np.argsort(ps)
    bh = ps[order] <= 0.05 * (np.arange(1, len(ps) + 1)) / len(ps)
    n_bh = int(bh.sum())
    n_raw = int((ps < 0.05).sum())

    best_obs = max(r["mean"] for r in rows)
    null = bootstrap_best_null(nets)
    p_emp = float((null >= best_obs).mean())

    print(f"\n  significantly profitable (one-sided):  {n_raw}/{len(rows)} raw, "
          f"{n_bh}/{len(rows)} after Benjamini-Hochberg")
    print(f"  best observed net/trade                : {best_obs:.1f}")
    print(f"  best-of-{len(rows)} under a recentred null    : "
          f"median {np.median(null):.1f}, 95th {np.percentile(null, 95):.1f}")
    print(f"  P(null best >= observed best)          : {p_emp:.3f}")

    gaps = [r["median_gap_days"] for r in rows]
    print(f"  median days between expiries           : {np.nanmedian(gaps):.0f}"
          f"   ({'weekly' if np.nanmedian(gaps) < 12 else 'monthly'} spacing)")

    return {"index": name, "cost_fraction": cost_frac, "configs": rows,
            "significant_raw": n_raw, "significant_bh": n_bh,
            "best_observed_mean": round(best_obs, 2),
            "null_best_median": round(float(np.median(null)), 2),
            "null_best_p95": round(float(np.percentile(null, 95)), 2),
            "p_null_ge_observed": round(p_emp, 4),
            "median_gap_days": float(np.nanmedian(gaps))}


def regime_split(name, cost_frac, dte=3):
    """BANKNIFTY changed expiry cadence mid-sample. Split on it and compare."""
    df = straddles(name, dte)
    if df.empty or len(df) < 30:
        return None
    weekly = df[df["gap_days"] <= 12]
    monthly = df[df["gap_days"] > 12]
    out = {}
    for label, sub in (("weekly", weekly), ("monthly", monthly)):
        r = evaluate(sub, cost_frac) if len(sub) >= 15 else None
        if r:
            out[label] = {k: round(v, 3) if isinstance(v, float) else v
                          for k, v in r.items() if k != "ci95"}
            print(f"    {label:<9}n={r['n']:>3}  net/trade {r['mean']:>7.1f}  "
                  f"t={r['t']:>5.2f}  p={r['p_one_sided']:.3f}  "
                  f"worst {r['worst']:.0f}")
        else:
            print(f"    {label:<9}too few expiries to test "
                  f"({len(sub)} < 15)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, default=0.10)
    ap.add_argument("--indices", nargs="*", default=["NIFTY", "BANKNIFTY"])
    a = ap.parse_args()

    out = []
    for name in a.indices:
        r = run(name, a.cost)
        if r:
            out.append(r)

    print(f"\n{'='*84}\nEXPIRY-REGIME SPLIT (3 DTE entry)\n{'='*84}")
    print("  NSE moved BANKNIFTY options from weekly to monthly mid-sample.")
    for name in a.indices:
        print(f"\n  {name}")
        reg = regime_split(name, a.cost)
        if reg:
            next(o for o in out if o["index"] == name)["regime_split"] = reg

    print(f"\n{'='*84}\nCOST SENSITIVITY (3 DTE entry)\n{'='*84}")
    print(f"  {'index':<12}" + "".join(f"{c*100:>12.0f}%" for c in COST_FRACTIONS))
    print("  " + "-" * 50)
    cost_rows = {}
    for name in a.indices:
        df = straddles(name, 3)
        if df.empty:
            continue
        vals = []
        for c in COST_FRACTIONS:
            r = evaluate(df, c)
            vals.append(r["mean"] if r else float("nan"))
        cost_rows[name] = [round(v, 2) for v in vals]
        print(f"  {name:<12}" + "".join(f"{v:>13.1f}" for v in vals))
    for o in out:
        o["cost_sensitivity"] = {"fractions": list(COST_FRACTIONS),
                                 "net_per_trade": cost_rows.get(o["index"], [])}

    d = paths.ROOT / "results"; d.mkdir(exist_ok=True)
    f = d / "vrp_test.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {f.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
