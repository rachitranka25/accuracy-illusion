#!/usr/bin/env python3
"""A strategy sweep, and the statistics that a strategy sweep requires.

Testing many strategies and reporting the best one is not research, it is
selection. With enough trials the best backtest is guaranteed to look good
whether or not anything works, and the standard errors quoted alongside it are
computed as if it had been the only thing tried.

This script does both halves of the job:

  1. THE SWEEP. A library of parameterised rules -- momentum, mean reversion,
     moving-average crosses, RSI, Bollinger, breakout, gap, volatility regime,
     time of day, day of week -- evaluated on non-overlapping k-bar holds with
     realistic costs, each with a first-half / second-half stability check.

  2. THE CORRECTION. Three treatments of the resulting multiplicity:

     Benjamini-Hochberg     controls the false discovery rate across the family
                            of tests, rather than pretending each was the first.

     Deflated Sharpe Ratio  (Bailey & Lopez de Prado, 2014) asks what Sharpe the
                            best of N independent trials would reach by luck
                            alone, and subtracts it. A strategy that does not
                            clear its own deflated threshold has not been shown
                            to work.

     PBO via CSCV           (Bailey, Borwein, Lopez de Prado & Zhu, 2017)
                            estimates the probability that the configuration
                            selected in-sample lands below median out-of-sample.
                            A PBO near 0.5 means the selection procedure itself
                            carries no information -- which is the honest
                            description of most published backtests.

Run:  python3 -m research.strategy_sweep [--horizon 30m] [--cost-bp 4]
"""
import argparse
import itertools
import json
from math import comb

import numpy as np
import pandas as pd
from scipy import stats

from oracle import paths
from research import common as C


# ══ signal library ════════════════════════════════════════════════════
def signals(bars):
    """Every rule returns a position in {-1, 0, +1}, using only past data."""
    c, h, l = bars["Close"], bars["High"], bars["Low"]
    out = {}

    def sgn(x):
        return np.sign(x).fillna(0)

    for n in (2, 3, 6, 9, 12, 18, 24, 36, 48, 72):
        out[f"momentum_{n}"] = sgn(c.pct_change(n))
        out[f"meanrev_{n}"] = -sgn(c.pct_change(n))

    for fast, slow in ((5, 20), (8, 16), (9, 21), (12, 26), (20, 50), (24, 72)):
        out[f"ema_cross_{fast}_{slow}"] = sgn(c.ewm(span=fast).mean() - c.ewm(span=slow).mean())

    for n in (7, 9, 14, 21, 28):
        rsi = _rsi(c, n)
        out[f"rsi_revert_{n}"] = pd.Series(
            np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0)), index=c.index)
        out[f"rsi_trend_{n}"] = sgn(rsi - 50)

    for n, k in itertools.product((10, 20, 40, 60), (1.0, 1.5, 2.0)):
        ma, sd = c.rolling(n).mean(), c.rolling(n).std()
        z = (c - ma) / (k * sd)
        out[f"bb_revert_{n}_{k}"] = pd.Series(
            np.where(z < -1, 1, np.where(z > 1, -1, 0)), index=c.index)
        out[f"bb_break_{n}_{k}"] = pd.Series(
            np.where(z > 1, 1, np.where(z < -1, -1, 0)), index=c.index)

    for n in (6, 12, 24, 48, 96):
        hi, lo = h.rolling(n).max().shift(), l.rolling(n).min().shift()
        out[f"breakout_{n}"] = pd.Series(
            np.where(c > hi, 1, np.where(c < lo, -1, 0)), index=c.index)
        out[f"fade_break_{n}"] = -out[f"breakout_{n}"]

    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    out["macd_hist"] = sgn(macd - macd.ewm(span=9).mean())
    out["macd_line"] = sgn(macd)

    day = c.index.normalize()
    prev_close = c.groupby(day).last().shift()          # previous session's close
    open_px = c.groupby(day).transform("first")
    gap = open_px / pd.Series(day.map(prev_close), index=c.index) - 1
    out["gap_follow"] = sgn(gap)
    out["gap_fade"] = -sgn(gap)

    ret = c.pct_change()
    vol = ret.rolling(24).std()
    hivol = (vol > vol.rolling(240).median()).astype(int)
    out["hivol_momentum"] = sgn(c.pct_change(6)) * hivol
    out["lovol_meanrev"] = -sgn(c.pct_change(6)) * (1 - hivol)

    mins = c.index.hour * 60 + c.index.minute
    out["morning_momentum"] = sgn(c.pct_change(6)) * ((mins < 660).astype(int))
    out["afternoon_momentum"] = sgn(c.pct_change(6)) * ((mins >= 810).astype(int))
    out["midday_fade"] = -sgn(c.pct_change(6)) * (((mins >= 660) & (mins < 810)).astype(int))

    for d in range(5):
        out[f"dow_{d}_long"] = pd.Series((c.index.dayofweek == d).astype(int), index=c.index)

    streak = sgn(ret).groupby((sgn(ret) != sgn(ret).shift()).cumsum()).cumcount() + 1
    out["streak_follow"] = sgn(ret) * (streak >= 3).astype(int)
    out["streak_fade"] = -sgn(ret) * (streak >= 3).astype(int)

    return {k: v.fillna(0).clip(-1, 1) for k, v in out.items()}


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# ══ evaluation ════════════════════════════════════════════════════════
def strategy_returns(pos, close, k, cost_bp):
    """Non-overlapping k-bar holds, costed on every position change.

    Sampling every k bars is what keeps each trade an independent observation;
    evaluating per bar would reintroduce exactly the inflation that
    research/overlap_simulation.py measures.
    """
    idx = np.arange(0, len(close) - k, k)
    p = pos.values[idx]
    fwd = (close.shift(-k) / close - 1).values[idx]

    day = close.index.normalize().values[idx]
    fwd = np.where(day == np.roll(day, -1), fwd, np.nan)     # no overnight holds

    turnover = np.abs(np.diff(np.concatenate([[0], p])))
    r = p * fwd - turnover * cost_bp / 1e4
    return pd.Series(r).dropna().values


def sharpe(r, periods_per_year):
    if len(r) < 20 or r.std() == 0:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


# ══ multiple-testing corrections ══════════════════════════════════════
def benjamini_hochberg(pvals, q=0.05):
    """Returns the BH-adjusted threshold and which hypotheses survive it."""
    p = np.asarray(pvals)
    order = np.argsort(p)
    n = len(p)
    thresh = q * (np.arange(1, n + 1)) / n
    passed = p[order] <= thresh
    if not passed.any():
        return 0.0, np.zeros(n, dtype=bool)
    cut = p[order][passed].max()
    return float(cut), p <= cut


def deflated_sharpe(sharpes, best_sr, n_obs, skew=0.0, kurt=3.0):
    """Bailey & Lopez de Prado (2014).

    Two parts. First the expected maximum Sharpe from N trials of pure noise,
    which is what selection alone buys you. Then the probability that the best
    observed Sharpe exceeds that benchmark given the sample's own length and
    non-normality.
    """
    sr = np.asarray([s for s in sharpes if s == s])
    n_trials = len(sr)
    if n_trials < 2 or n_obs < 20:
        return float("nan"), float("nan")

    var_sr = sr.var(ddof=1)
    e = 0.5772156649                                    # Euler-Mascheroni
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    sr0 = np.sqrt(var_sr) * ((1 - e) * z1 + e * z2)     # expected max under the null

    denom = np.sqrt(1 - skew * best_sr + (kurt - 1) / 4 * best_sr ** 2)
    if denom <= 0:
        return float(sr0), float("nan")
    dsr = stats.norm.cdf((best_sr - sr0) * np.sqrt(n_obs - 1) / denom)
    return float(sr0), float(dsr)


def empirical_null_max_sharpe(P, fwd, cost_bp, n_boot, rng, block=10):
    """What does the best of N strategies look like when nothing works?

    The parametric Deflated Sharpe answers this with a formula that assumes the
    trial Sharpes are drawn from a common normal. That assumption fails here:
    the strategies differ enormously in turnover, so their Sharpe dispersion is
    driven by real cost differences rather than by noise, and the formula's null
    benchmark becomes meaningless.

    So the null is measured instead. A circular block bootstrap resamples the
    forward returns in blocks -- preserving their autocorrelation and fat tails
    -- while leaving every strategy's positions and turnover untouched. That
    destroys any relationship between signal and outcome and nothing else. The
    maximum Sharpe across all strategies is recorded each iteration, giving the
    distribution of "best backtest of N" under a true null.
    """
    n_obs, n_str = P.shape
    turnover = np.abs(np.diff(np.vstack([np.zeros(n_str), P]), axis=0))
    cost = turnover * cost_bp / 1e4

    n_blocks = int(np.ceil(n_obs / block))
    maxes = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n_obs, n_blocks)
        idx = np.concatenate([np.arange(st, st + block) % n_obs for st in starts])[:n_obs]
        r = P * fwd[idx][:, None] - cost
        mu, sd = r.mean(0), r.std(0)
        maxes[b] = np.nanmax(np.where(sd > 0, mu / np.where(sd > 0, sd, 1), np.nan))
    return maxes


def pbo_cscv(returns_matrix, n_splits=10):
    """Probability of Backtest Overfitting, combinatorially symmetric CV.

    The sample is cut into S blocks. For every way of choosing S/2 blocks as
    "in-sample", the best strategy there is found and its rank among all
    strategies on the remaining blocks is recorded. PBO is the share of splits
    where that winner lands in the bottom half out-of-sample.

    PBO near 0.5 says selecting on in-sample performance tells you nothing about
    out-of-sample performance.
    """
    M = np.asarray(returns_matrix)                      # (observations, strategies)
    n_obs, n_str = M.shape
    if n_str < 2 or n_obs < n_splits * 4:
        return float("nan"), 0

    blocks = np.array_split(np.arange(n_obs), n_splits)
    half = n_splits // 2
    logits = []

    for combo in itertools.combinations(range(n_splits), half):
        is_idx = np.concatenate([blocks[b] for b in combo])
        oos_idx = np.concatenate([blocks[b] for b in range(n_splits) if b not in combo])

        is_sr = np.nan_to_num(M[is_idx].mean(0) / (M[is_idx].std(0) + 1e-12))
        oos_sr = np.nan_to_num(M[oos_idx].mean(0) / (M[oos_idx].std(0) + 1e-12))

        best = int(np.argmax(is_sr))
        rank = float(stats.rankdata(oos_sr)[best]) / (n_str + 1)
        logits.append(np.log(rank / (1 - rank)) if 0 < rank < 1 else 0.0)

    logits = np.array(logits)
    return float((logits <= 0).mean()), len(logits)


# ══ driver ════════════════════════════════════════════════════════════
def run(name, horizon, cost_bp, folds_pbo, n_boot):
    k = C.HORIZON_BARS[horizon]
    bars = C.load_bars(name)
    close = bars["Close"]
    lib = signals(bars)

    bars_per_year = 75 * 250 / k

    # Shared sampling grid: one observation per k bars, no overnight holds.
    grid = np.arange(0, len(close) - k, k)
    fwd_all = (close.shift(-k) / close - 1).values[grid]
    day = close.index.normalize().values[grid]
    valid = (day == np.roll(day, -1)) & np.isfinite(fwd_all)
    grid, fwd = grid[valid], fwd_all[valid]

    rows, series, pos_cols = [], {}, {}

    for label, pos in lib.items():
        r = strategy_returns(pos, close, k, cost_bp)
        if len(r) < 100 or np.allclose(r, 0):
            continue
        h = len(r) // 2
        sr = sharpe(r, bars_per_year)
        # One-sided: the hypothesis is "this makes money", not "this differs
        # from zero". A reliably losing strategy is not a discovery.
        t, p_two = stats.ttest_1samp(r, 0.0)
        p_profit = p_two / 2 if t > 0 else 1 - p_two / 2
        rows.append({
            "strategy": label, "n": len(r), "sharpe": sr,
            "sharpe_per_period": float(r.mean() / r.std()) if r.std() else float("nan"),
            "sharpe_h1": sharpe(r[:h], bars_per_year),
            "sharpe_h2": sharpe(r[h:], bars_per_year),
            "mean_bp": r.mean() * 1e4, "t": float(t), "p": float(p_profit),
        })
        series[label] = r
        pos_cols[label] = pos.values[grid]

    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    df["stable"] = (np.sign(df.sharpe_h1) == np.sign(df.sharpe_h2)) & (df.sharpe > 0)

    print(f"\n{'='*84}\n{name} — {len(df)} strategies, {horizon} holds, "
          f"{cost_bp} bp round-trip cost\n{'='*84}")
    print(f"{'strategy':<24}{'Sharpe':>8}{'H1':>7}{'H2':>7}{'bp/trade':>10}"
          f"{'t':>7}{'p':>9}  stable")
    print("-" * 84)
    for _, r in df.head(10).iterrows():
        print(f"{r.strategy:<24}{r.sharpe:>8.2f}{r.sharpe_h1:>7.2f}{r.sharpe_h2:>7.2f}"
              f"{r.mean_bp:>10.2f}{r.t:>7.2f}{r.p:>9.4f}  {'yes' if r.stable else 'no'}")
    print(f"... {len(df)-10} more")

    # ── corrections ────────────────────────────────────────────────────
    naive = int((df.p < 0.05).sum())
    bh_cut, bh_pass = benjamini_hochberg(df.p.values, 0.05)
    bonf = int((df.p < 0.05 / len(df)).sum())

    n_obs = int(df.n.median())
    best = df.iloc[0]
    best_r = series[best.strategy]
    # Bailey & Lopez de Prado work in the sample's own periodicity. Feeding
    # annualised Sharpes here inflates the null benchmark by sqrt(periods/yr)
    # and makes every strategy fail for the wrong reason.
    sr0_p, dsr = deflated_sharpe(
        df.sharpe_per_period.values, float(best.sharpe_per_period), n_obs,
        skew=float(stats.skew(best_r)),
        kurt=float(stats.kurtosis(best_r, fisher=False)))
    ann = np.sqrt(bars_per_year)
    sr0 = sr0_p * ann

    common_n = min(len(v) for v in series.values())
    M = np.column_stack([v[-common_n:] for v in series.values()])
    pbo, n_combos = pbo_cscv(M, folds_pbo)

    # empirical null: best-of-N under block-bootstrapped returns
    rng = np.random.default_rng(C.SEED)
    P = np.column_stack([pos_cols[l] for l in df.strategy])
    null_max = empirical_null_max_sharpe(P, fwd, cost_bp, n_boot, rng)
    ann = np.sqrt(bars_per_year)
    null_max_ann = null_max * ann
    obs_best = float(best.sharpe)
    p_emp = float((null_max_ann >= obs_best).mean())

    n_pos = int((df.sharpe > 0).sum())
    print(f"\n  MULTIPLE TESTING ({len(df)} trials, one-sided test for profit)")
    print(f"    positive Sharpe at all                 : {n_pos} of {len(df)}")
    print(f"    significantly profitable, uncorrected  : {naive} strategies")
    print(f"    ... surviving Bonferroni (p<{0.05/len(df):.5f})    : {bonf} strategies")
    print(f"    ... surviving Benjamini-Hochberg FDR 5%: {int(bh_pass.sum())} strategies")
    print(f"    positive AND stable across both halves : {int(df.stable.sum())} strategies")

    print(f"\n  DEFLATED SHARPE (Bailey & Lopez de Prado 2014)")
    print(f"    best observed Sharpe (annualised)      : {best.sharpe:6.2f}  ({best.strategy})")
    print(f"    expected max Sharpe from {len(df):>3} null trials: {sr0:6.2f}  "
          f"<- what selection alone buys")
    print(f"    deflated Sharpe probability            : {dsr:6.3f}")
    print(f"    NOTE: the parametric benchmark above assumes the trial Sharpes are")
    print(f"    a common-variance normal sample. Here their dispersion is driven by")
    print(f"    turnover differences rather than noise, so the formula overstates")
    print(f"    the null. The block-bootstrap null below is the one to read.")

    print(f"\n  EMPIRICAL NULL (block bootstrap, {n_boot} resamples)")
    print(f"    best observed Sharpe                   : {obs_best:6.2f}")
    print(f"    best-of-{len(df)} Sharpe under the null     : "
          f"median {np.median(null_max_ann):.2f}, "
          f"p95 {np.percentile(null_max_ann,95):.2f}, "
          f"max {null_max_ann.max():.2f}")
    print(f"    P(null best >= observed best)          : {p_emp:6.3f}"
          f"   {'-> the winner is what selection alone produces' if p_emp > 0.05 else '-> the winner exceeds selection luck'}")

    print(f"\n  PROBABILITY OF BACKTEST OVERFITTING (CSCV, {n_combos} splits)")
    print(f"    PBO                                    : {pbo:6.3f}")
    if naive == 0:
        print(f"    (degenerate: nothing is significantly profitable in-sample, so")
        print(f"     there is nothing for the selection procedure to overfit to)")
    elif pbo > 0.4:
        print(f"    -> in-sample selection carries no out-of-sample information")
    else:
        print(f"    -> the in-sample winner does tend to rank above median OOS")

    return {"index": name, "horizon": horizon, "cost_bp": cost_bp,
            "n_strategies": len(df), "n_obs_median": n_obs,
            "best": {"strategy": best.strategy, "sharpe": round(float(best.sharpe), 3),
                     "p": round(float(best.p), 5)},
            "significant_uncorrected": naive, "significant_bonferroni": bonf,
            "significant_bh_fdr5": int(bh_pass.sum()),
            "stable_and_positive": int(df.stable.sum()),
            "expected_max_sharpe_null_annualised": round(sr0, 3),
            "n_positive_sharpe": n_pos,
            "deflated_sharpe_prob": round(dsr, 4),
            "pbo": round(pbo, 4), "pbo_splits": n_combos,
            "empirical_null": {"boot": n_boot,
                               "median_max_sharpe": round(float(np.median(null_max_ann)), 3),
                               "p95_max_sharpe": round(float(np.percentile(null_max_ann, 95)), 3),
                               "p_observed_vs_null": round(p_emp, 4)},
            "table": df.round(4).to_dict("records")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="30m", choices=list(C.HORIZON_BARS))
    ap.add_argument("--cost-bp", type=float, default=4.0,
                    help="round-trip cost in basis points (~4bp = futures + slippage)")
    ap.add_argument("--pbo-splits", type=int, default=10)
    ap.add_argument("--boot", type=int, default=500,
                    help="block-bootstrap resamples for the empirical null")
    ap.add_argument("--indices", nargs="*", default=C.INDICES)
    a = ap.parse_args()

    out = [run(n, a.horizon, a.cost_bp, a.pbo_splits, a.boot) for n in a.indices]

    d = paths.ROOT / "results"; d.mkdir(exist_ok=True)
    f = d / f"strategy_sweep_{a.horizon}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {f.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
