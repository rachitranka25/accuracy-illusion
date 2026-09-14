# 11 — Testing 86 strategies, and what that does to the statistics

**Claim tested.** That the best strategy out of a large sweep is a finding.

**Verdict.** It is not, and this is measurable rather than a matter of opinion.
On BANKNIFTY the best of 86 strategies reaches an annualised Sharpe of 0.21.
A block-bootstrap null — the same 86 strategies run on resampled returns with
every relationship to the signal destroyed — produces a best-of-86 Sharpe with
**median 0.44 and 95th percentile 1.31**. The observed winner is *worse* than
what pure selection typically manufactures.

`P(null best ≥ observed best) = 0.658.`

> Status: **REPRODUCIBLE** — `research/strategy_sweep.py`.

---

## The sweep

86 parameterised rules on 17 months of 5-minute bars: momentum and mean
reversion at ten lookbacks, six EMA crossover pairs, RSI trend and reversion at
five periods, twelve Bollinger configurations, breakout and fade at five
windows, MACD, gap follow and fade, volatility-regime conditioning, three
time-of-day variants, day-of-week, and streak follow/fade.

Evaluated on **non-overlapping 30-minute holds** with 4 bp round-trip cost, no
overnight positions, and a first-half / second-half stability check. The
non-overlapping sampling is not incidental — scoring these per bar would
reintroduce exactly the inflation measured in
[finding 02](02-measurement-illusions.md).

## What came back

| | BANKNIFTY | NIFTY 50 |
|---|---|---|
| Strategies evaluated | 86 | 86 |
| **Positive Sharpe at all** | **2 of 86** | 2 of 86 |
| Significantly profitable (one-sided, uncorrected) | **0** | 0 |
| Surviving Bonferroni | 0 | 0 |
| Surviving Benjamini-Hochberg FDR 5% | 0 | 0 |
| Positive **and** stable across both halves | 1 | 1 |
| Best Sharpe | 0.21 (`dow_2_long`) | 0.73 (`dow_0_long`) |

The two survivors are day-of-week rules — "be long on Wednesday", "be long on
Monday". They are in the library precisely as controls: rules with no mechanism,
included so that when they win, the win is legible as noise.

A methodological note on the test direction. An earlier version of this sweep
used a two-sided t-test and reported "42 of 86 significant", which is true and
useless: those are strategies that reliably *lose*. The hypothesis is
"this makes money", so the test is one-sided. Reliable losers are not
discoveries, they are the cost of trading noise.

## Why a parametric correction was not enough

The Deflated Sharpe Ratio (Bailey & López de Prado, 2014) is the standard tool:
estimate what the maximum Sharpe of *N* trials would be under a null, and
require the observed best to exceed it. Applied here it returns an expected null
maximum of **7.77** annualised, which is obviously wrong.

The reason is that DSR estimates the null from the cross-sectional variance of
the trial Sharpes, assuming they are a common-variance normal sample. In this
library they are not. The strategies differ enormously in turnover — a rule that
trades every bar pays cost on every bar, a day-of-week rule trades once a week —
so their Sharpe dispersion is dominated by real, deterministic cost differences
rather than by sampling noise. Feeding that dispersion into the formula inflates
the null benchmark past any achievable value.

This is reported rather than quietly dropped, because "we applied the standard
correction and it gave a nonsense number" is itself a finding about applying
these tools to heterogeneous strategy libraries.

## The empirical null

So the null was measured instead of assumed.

A **circular block bootstrap** (block length 10 observations) resamples the
forward returns, preserving their autocorrelation and fat tails, while leaving
every strategy's positions and turnover exactly as they are. This destroys the
relationship between signal and outcome and changes nothing else. The maximum
Sharpe across all 86 strategies is recorded, 500 times.

| | BANKNIFTY |
|---|---|
| Best observed Sharpe | **0.21** |
| Null best-of-86, median | 0.44 |
| Null best-of-86, 95th percentile | 1.31 |
| Null best-of-86, maximum seen | 2.73 |
| **P(null ≥ observed)** | **0.658** |

Read plainly: if nothing in this library worked at all, the sweep would
typically report a better winner than the one it actually found.

This construction is preferred to the parametric version for three reasons — it
makes no distributional assumption about the trial Sharpes, it preserves each
strategy's real turnover and cost structure, and it preserves the serial
dependence of the return series.

## Probability of backtest overfitting

PBO via combinatorially symmetric cross-validation (Bailey, Borwein, López de
Prado & Zhu, 2017) was also computed across 252 splits. It is reported as
**degenerate here**: with nothing significantly profitable in-sample, there is
no selection for the procedure to overfit to, and the resulting number carries
no information. It is retained in the output so that the same script produces a
meaningful PBO if it is ever run on a library where something does work.

Reporting a degenerate statistic as though it were a result would be the same
class of error this document is about.

## The general point

The arithmetic is not specific to trading. Test 86 things at the 5% level and
about four will pass by construction. Test 100+ configurations across several
indices and horizons — which is what this project did over fourteen months, and
what most published strategy research does — and the best result is drawn from
the tail of a null distribution, not from the body of a real one.

Three corrections make the difference between a sweep and a finding:

1. **A one-sided test**, so that losing reliably is not counted as significant.
2. **An empirical null that preserves the structure of the data**, so the
   comparison is against what selection actually produces rather than what a
   formula assumes it produces.
3. **Reporting the whole distribution of trials**, not the maximum.

Applied to this project's own sweep, all three point the same way: nothing here
beats selection luck.
