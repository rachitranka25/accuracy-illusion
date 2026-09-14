# 01 — The directional ceiling

**Claim tested.** That a sufficiently good model can predict the direction of
the next 5–60 minutes on BANKNIFTY / NIFTY 50 from freely available 5-minute
OHLC data.

**Verdict.** It cannot. The limit is not the model. The information is not in
the data.

All numbers below are **REPRODUCIBLE** —
[`research/ceiling_test.py`](../research/ceiling_test.py) and
[`research/model_zoo.py`](../research/model_zoo.py), with raw output in
[`results/`](../results/).

---

## The ceiling test

The usual way to argue about model quality is to try more models. That argument
never ends, because there is always one more architecture. So the question was
asked in a form that does not depend on the model at all.

An unconstrained random forest was fitted to the live system's own feature set
until it memorised the training data, on a purged expanding-window walk-forward
with five folds. 30-minute horizon, ~24,000 bars per index, ~16,850 out-of-sample
predictions.

### BANKNIFTY

| Measurement | Train | Out-of-sample | 95% CI | p vs 0.5 |
|---|---|---|---|---|
| Over-capacity model, **real labels** | 100.0% | **50.5%** | [49.7, 51.2] | 0.221 |
| Over-capacity model, **shuffled labels** | 100.0% | **50.2%** | [49.4, 51.0] | 0.606 |
| 1-NN oracle | — | 48.8% | [48.1, 49.6] | 0.002 |
| Majority-class baseline | — | 49.3% | [48.5, 50.0] | 0.061 |

### NIFTY 50

| Measurement | Train | Out-of-sample | 95% CI | p vs 0.5 |
|---|---|---|---|---|
| Over-capacity model, **real labels** | 100.0% | **50.0%** | [49.2, 50.7] | 0.914 |
| Over-capacity model, **shuffled labels** | 100.0% | **50.2%** | [49.5, 51.0] | 0.528 |
| 1-NN oracle | — | 50.2% | [49.4, 50.9] | 0.633 |
| Majority-class baseline | — | 49.8% | [49.1, 50.6] | 0.655 |

**The shuffled-label row is the decisive one.** The model memorises the training
set perfectly and generalises at chance. Randomly permuting the labels —
destroying any link between features and outcome — changes out-of-sample
accuracy by **0.3 points on both indices**. The pipeline performs identically on
real data and on noise, which means it was reading noise all along.

Neither real-label result is distinguishable from a coinflip at any conventional
threshold.

## The positive control

A null result is only meaningful if the pipeline could have detected a signal
had one been present. Otherwise "we found nothing" is indistinguishable from
"our code is broken" — an objection that applies to a great deal of published
negative work, and the reason this control exists.

A synthetic feature was injected, agreeing with the label on a known fraction of
rows and random on the rest, so the recoverable accuracy is known in advance:

| Injected signal strength | Recoverable | BANKNIFTY | NIFTY 50 |
|---|---|---|---|
| 0.10 | ~55% | **52.4%** | **52.0%** |
| 0.30 | ~65% | **63.5%** | **63.7%** |

The pipeline recovers both, at p < 0.001. It finds signal when signal exists. It
finds nothing on real features because there is nothing there.

## Serial structure — and a warning built into the measurement

Autocorrelation of the 30-minute forward return, measured two ways:

| | lag 1 | lag 3 | lag 6 | lag 12 | n |
|---|---|---|---|---|---|
| **Overlapping** (every bar) | **0.841** | 0.523 | 0.042 | 0.021 | 24,074 |
| **Non-overlapping** (every 6th bar) | 0.067 | 0.019 | 0.009 | −0.027 | 4,013 |

The overlapping series looks strongly autocorrelated. It is not information —
it is arithmetic. Two 6-bar forward returns measured one bar apart share five of
their six bars. The correlation decays to zero at **exactly lag 6**, which is
the horizon length, and that is the signature of overlap rather than structure.

Sampled so that no two observations share a bar, the autocorrelation is
essentially zero.

This single table is the paper's central mechanism visible in one place, and it
is developed in [finding 02](02-measurement-illusions.md) and quantified in
[finding 11](11-multiple-testing.md).

## Every model family, same answer

Eight families plus a baseline, trained on the identical task, the identical
features, the identical purged walk-forward splits, each with its own
shuffled-label control. Full table in
[`results/model_zoo_30m.json`](../results/model_zoo_30m.json).

Families: logistic regression, gradient boosting, LightGBM, XGBoost,
k-nearest-neighbours, LSTM, Transformer, and a compact Temporal Fusion
Transformer (variable-selection network, GRN gating, interpretable multi-head
attention over an LSTM encoder — the components that define the architecture,
without static covariates or a quantile head, which this task does not have).

The result is that architecture choice does not matter on this problem: every
family lands within a couple of points of 50%, and each one scores the same on
shuffled labels as on real ones.

XGBoost is worth a separate note. It was included because a widely circulated
paper reports ~71% directional accuracy with it. Reproduced on NSE index data it
returns chance. The difference is not the algorithm — it is a less efficient
market and a selectively sampled high-volatility subset.

## Why the features cannot help

The live engine's feature set — 1-bar return, RSI, EMA distance, MACD histogram,
Bollinger width and %b, ATR%, Parkinson volatility, VWAP distance, time-of-day,
day-of-week, regime volatility, Hurst exponent, distance from the previous
session's high and low — is fifteen columns computed from **one** underlying
series. They are transforms of price. Reshaping price cannot reveal something
price did not already contain.

This is the reasoning that motivated the move to open interest and participant
positioning in [finding 06](06-open-interest.md): not a better model, a
genuinely different information source.

## The rule-based sweep

Separately from the model work, 86 parameterised strategies were evaluated on
non-overlapping holds with costs. Two had a positive Sharpe; none was
significantly profitable; and the best of them was worse than what a
block-bootstrap null produces from selection alone. See
[finding 11](11-multiple-testing.md).

## What the literature says

This matches the published consensus rather than contradicting it. Peer-reviewed
work on liquid index intraday prediction reports 50–55% as normal and 55–58% as
a realistic ceiling; results above 60% are rare and heavily caveated. In review
papers, claimed accuracies of 80–90% are treated as a **diagnostic of lookahead
bias**, not as an achievement.

The genuine short-horizon edge documented in the literature lives in order-flow
and limit-order-book microstructure — tick-level Level-2 data with colocated
execution. That is a data and latency gap, not an intelligence gap, and it is
not reachable from a retail account. Our own live order-flow collector
([finding 09](09-data-assets.md)) exists because that is the only direction left
that is not already closed.

## Consequence

Everything downstream in this project follows from this finding:

1. Direction is not the product. It is the control condition.
2. The question changes from *which way* to *how far* → [finding 04](04-volatility-predictability.md).
3. Anything reporting high directional accuracy is assumed to be a measurement
   error until proven otherwise → [finding 02](02-measurement-illusions.md).
