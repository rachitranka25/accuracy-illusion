"""Shared evaluation harness for the predictability experiments.

Every experiment in this directory that reports a directional accuracy goes
through this file, so that "the LSTM got 49%" and "XGBoost got 50%" are
statements about the same data, the same features, the same splits and the same
labels. Differences between models are then differences between models.

Three things here are load-bearing and are the reason the numbers can be
trusted:

1. **Purging.** A k-bar-ahead label overlaps the next k-1 observations. Without
   removing them from around the split boundary, training data leaks into test
   through the label, which is the single most common way a backtest lies.

2. **The shuffled-label control.** Every experiment is also run with the labels
   randomly permuted. If the real-label score and the shuffled score are the
   same, the pipeline was reading noise. This is a stronger statement than any
   train/test gap, because it does not depend on the model being well specified.

3. **Interval, not point.** Accuracy is reported with an exact binomial
   confidence interval and a p-value against 0.5. A single number without a
   sample size is not evidence, and most of the accuracies in this field are
   quoted without one.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from oracle import paths
from oracle.models import direction as fe

HORIZON_BARS = {"5m": 1, "15m": 3, "30m": 6, "60m": 12}
INDICES = ["BANKNIFTY", "NIFTY50"]
SEED = 0


# ── data ───────────────────────────────────────────────────────────────
def load_bars(name):
    """5-minute bars for one index, session hours only."""
    f = paths.CACHE / f"{name}_5m_long.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} missing — rebuild with: python3 -m oracle.data.index_history {name}")
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    t = df.index.time
    keep = (t >= pd.Timestamp("09:15").time()) & (t <= pd.Timestamp("15:30").time())
    return df[keep].sort_index()


def dataset(name, horizon="30m"):
    """Features + binary direction label, aligned and cleaned.

    Features come from the live engine's own builder, so the benchmark measures
    the production feature set rather than a convenient stand-in.
    """
    k = HORIZON_BARS[horizon]
    bars = load_bars(name)
    feats, close = fe.build5(bars)
    X = feats[fe.BASE_COLS].copy()

    # Label: did price rise over the next k bars? Never crosses a session
    # boundary -- an overnight gap is not what an intraday model is asked for.
    fwd = close.shift(-k) / close - 1
    same_day = pd.Series(close.index.normalize(), index=close.index)
    fwd[same_day.shift(-k).values != same_day.values] = np.nan

    ok = X.notna().all(axis=1) & fwd.notna() & (fwd != 0)
    return X[ok], (fwd[ok] > 0).astype(int).values, close[ok]


# ── splits ─────────────────────────────────────────────────────────────
def walk_forward(n, k, folds=5, min_train=0.30):
    """Expanding-window walk-forward with purge + embargo of k bars.

    Yields (train_idx, test_idx). Training always precedes testing in time, and
    the k bars whose labels overlap the test window are dropped from the end of
    the training set.
    """
    start = int(n * min_train)
    edges = np.linspace(start, n, folds + 1).astype(int)
    for i in range(folds):
        te0, te1 = edges[i], edges[i + 1]
        tr_end = max(0, te0 - k)          # purge the overlapping labels
        if tr_end < 100 or te1 - te0 < 50:
            continue
        yield np.arange(0, tr_end), np.arange(te0, te1)


# ── metrics ────────────────────────────────────────────────────────────
@dataclass
class Score:
    """An accuracy that carries its own uncertainty."""
    name: str
    correct: int = 0
    n: int = 0
    train_acc: float = float("nan")
    extra: dict = field(default_factory=dict)

    @property
    def acc(self):
        return self.correct / self.n if self.n else float("nan")

    @property
    def ci(self):
        """Exact (Clopper-Pearson) 95% interval."""
        if not self.n:
            return (float("nan"), float("nan"))
        lo, hi = stats.binomtest(self.correct, self.n).proportion_ci(0.95)
        return lo, hi

    @property
    def p_vs_coin(self):
        """Two-sided exact p-value against the 0.5 null."""
        if not self.n:
            return float("nan")
        return stats.binomtest(self.correct, self.n, 0.5).pvalue

    def add(self, y_true, y_pred):
        self.correct += int((np.asarray(y_true) == np.asarray(y_pred)).sum())
        self.n += len(y_true)

    def row(self):
        lo, hi = self.ci
        return (f"{self.name:<26}{self.acc*100:7.1f}%  "
                f"[{lo*100:.1f}, {hi*100:.1f}]  "
                f"p={self.p_vs_coin:<8.3f} n={self.n:>7,}"
                + (f"  train={self.train_acc*100:.1f}%"
                   if self.train_acc == self.train_acc else ""))

    def as_dict(self):
        lo, hi = self.ci
        return {"model": self.name, "accuracy": round(self.acc, 4),
                "ci95": [round(lo, 4), round(hi, 4)],
                "p_vs_coinflip": round(self.p_vs_coin, 4), "n": self.n,
                "train_accuracy": (round(self.train_acc, 4)
                                   if self.train_acc == self.train_acc else None),
                **self.extra}


# ── formal tests of directional predictability ─────────────────────────
def pesaran_timmermann(y_true, y_pred):
    """Pesaran-Timmermann (1992) test of directional accuracy.

    The right null for this problem. A plain binomial test against 0.5 assumes
    the two classes are balanced; PT compares the realised hit-rate against the
    rate expected if the forecast and the outcome were *independent* given their
    own marginal rates. A model that always predicts UP in a market that drifts
    up beats 0.5 without predicting anything, and PT is what catches that.

    Returns (statistic, two-sided p-value). Statistic is asymptotically N(0,1).
    """
    y, f = np.asarray(y_true), np.asarray(y_pred)
    n = len(y)
    if n < 30:
        return float("nan"), float("nan")

    p_hit = (y == f).mean()
    py, pf = y.mean(), f.mean()
    p_star = py * pf + (1 - py) * (1 - pf)          # hit-rate under independence

    var_hit = p_star * (1 - p_star) / n
    var_star = (((2 * py - 1) ** 2) * pf * (1 - pf) / n
                + ((2 * pf - 1) ** 2) * py * (1 - py) / n
                + 4 * py * pf * (1 - py) * (1 - pf) / n ** 2)
    denom = var_hit - var_star
    if denom <= 0:
        return float("nan"), float("nan")

    stat = (p_hit - p_star) / np.sqrt(denom)
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))


def diebold_mariano(err_a, err_b, h=1):
    """Diebold-Mariano test that two forecasts have equal expected loss.

    Uses squared error and a Newey-West variance with h-1 lags, because
    h-step-ahead forecast errors are serially correlated by construction.
    Returns (statistic, two-sided p-value); negative favours A.
    """
    d = np.asarray(err_a) ** 2 - np.asarray(err_b) ** 2
    n = len(d)
    if n < 30:
        return float("nan"), float("nan")

    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0
    for lag in range(1, h):
        g = np.mean((d[lag:] - dbar) * (d[:-lag] - dbar))
        var += 2 * (1 - lag / h) * g
    if var <= 0:
        return float("nan"), float("nan")

    stat = dbar / np.sqrt(var / n)
    return stat, 2 * (1 - stats.norm.cdf(abs(stat)))


# ── scoring a hit sequence honestly ────────────────────────────────────
def hit_runs(h):
    """Lengths of maximal constant stretches in a hit indicator sequence."""
    h = np.asarray(h)
    if len(h) == 0:
        return np.array([1])
    breaks = np.flatnonzero(np.diff(h)) + 1
    return np.diff(np.concatenate([[0], breaks, [len(h)]]))


def effective_n(h):
    """Design effect for a run-structured hit sequence (Kish; Moulton 1990).

    Clusters are the runs of constant hit indicator, so
        n_eff = n / (sum_i L_i^2 / n).
    Reported alongside n whenever forecasts overlap.
    """
    L = hit_runs(h)
    n = int(L.sum())
    if n < 2:
        return float(n), 1.0
    deff = max(1.0, float((L ** 2).sum()) / n)
    return n / deff, deff


def interval_at(correct, n_eff, z=1.96):
    """Normal interval on a proportion, evaluated at an effective sample size."""
    n = max(n_eff, 2.0)
    p = correct
    se = np.sqrt(max(p * (1 - p), 1e-12) / n)
    return max(0.0, p - z * se), min(1.0, p + z * se)


# ── one generative model, shared by every simulation ───────────────────
#
# An earlier version of this work used two different synthetic forecasters in
# two different experiments, and they disagreed about the effective sample size
# of the same session by a factor of four. Both now come from here.
#
# The model: a forecaster holds a direction for a run drawn from the empirically
# measured run-length distribution, and picks that direction to agree with the
# market at the start of the run with probability `bias`. Crucially the CALL is
# constant within a run while the TRUTH is not, so the hit indicator is only
# partly block-structured. A forecaster whose hits were constant within a run
# would be a different and much more strongly dependent object, and assuming one
# inflates the correction.

def persistent_forecast(truth, run_bars, rng, bias=0.5):
    """Calls that persist, with a tunable tilt toward being right.

    bias = 0.5 gives a forecaster with no skill whatsoever. Higher values tilt
    each run toward the direction prevailing when the run starts, which is how a
    real model with a small edge behaves.
    """
    truth = np.asarray(truth)
    n = len(truth)
    out = np.empty(n, dtype=int)
    i = 0
    while i < n:
        length = min(int(rng.choice(run_bars)), n - i)
        anchor = truth[i]
        call = anchor if rng.random() < bias else 1 - anchor
        out[i:i + length] = call
        i += length
    return out


def true_hit_rate(sessions, run_bars, bias, rng, reps=400):
    """The long-run hit-rate of the above DGP, measured rather than assumed.

    With persistence the realised rate is not `bias`, because the call is
    anchored on one bar and then held across others. Coverage has to be tested
    against the rate the process actually produces.
    """
    hit = tot = 0
    for t in sessions:
        for _ in range(reps):
            p = persistent_forecast(t, run_bars, rng, bias)
            hit += int((p == t).sum()); tot += len(t)
    return hit / tot


def header(title, sub=""):
    print(f"\n{'='*78}\n{title}")
    if sub:
        print(sub)
    print("=" * 78)
    print(f"{'model':<26}{'OOS acc':>8}  {'95% CI':^14}  {'p vs 0.5':<10} {'n':>7}")
    print("-" * 78)


def shuffled(y, rng):
    """Labels permuted. The control that makes a null result interpretable."""
    out = y.copy()
    rng.shuffle(out)
    return out
