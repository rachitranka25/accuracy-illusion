# 12 — The correction: effective sample size

**Claim tested.** That the inflation documented in
[finding 02](02-measurement-illusions.md) can be corrected, rather than only
diagnosed.

**Verdict.** Partly. A nominal 95 % interval on a session hit-rate actually
covers **75 %** of the time. HAC and a moving-block bootstrap each reach ~87 %.
The run-structure design effect **over**-covers at 99.5 %, so it is a
conservative bound rather than an exact correction, and we report it as one.

The number that matters is the size of n_eff: a session reporting 69 forecasts
carries roughly **12** independent observations, not 69.

> An earlier version of this finding reported 40 % naive coverage and 94 % for
> the corrected estimator. Those came from a synthetic forecaster whose *hits*
> were constant within a run, which is not how a real forecaster behaves — the
> call persists, the market does not. Both simulations now share one generative
> model ([`research/common.py`](../research/common.py)), and the numbers above
> are from it. The discrepancy was found by external audit, not by us.

> Status: **REPRODUCIBLE** — `research/effective_sample.py`,
> [`results/effective_sample_30m.json`](../results/effective_sample_30m.json).

---

## The closed form

Let `h_t = 1{ŷ_t = y_t}` be the hit indicator of a forecaster issuing a
*k*-bar-ahead call every bar, and `p̂ = (1/n) Σ h_t` the reported hit-rate.
Practitioners quote `Var(p̂) = p(1−p)/n`, which holds only for independent `h_t`.
In general

```
Var(p̂) = p(1−p)/n · [ 1 + 2 Σ_j (1 − j/n) ρ_j ],    ρ_j = Corr(h_t, h_{t+j})
```

and dividing *n* by the bracketed factor gives the effective sample size.

Two structures drive `ρ_j`, and both are required. The outcome is autocorrelated
out to lag *k* because consecutive windows share bars. The prediction persists
in runs. Redraw the prediction each bar and `h_t` becomes i.i.d. whatever the
outcome does — every `ρ_j` vanishes and `n_eff = n`.

Because the dependence here is **block-shaped** rather than smoothly decaying,
the sum has a direct form. If the sequence decomposes into runs of length `L_i`
within which `h_t` is constant, conditioning on the lengths gives
`Var(p̂) = p(1−p) Σ L_i² / n²`, so

```
n_eff = n / ( Σ_i L_i² / n )  →  n · E[L] / E[L²]
```

**The inflation is governed by `E[L²]`, not `E[L]`.** A heavy-tailed run-length
distribution inflates the variance far past what its mean suggests — and that is
exactly the regime a real forecaster occupies. Our production log has a mean run
of 15.2 minutes and a longest run of 135.

## Validation by coverage

An interval is judged by whether it covers. We generate a forecaster whose true
skill is fixed by construction, with calls persisting per the measured
run-length distribution, and count how often a nominal 95 % interval contains
that known truth.

BANKNIFTY, 30-minute horizon, nominal n = 69. NIFTY 50 agrees within 1 point on
every entry.

| Estimator | 50 % | 55 % | 60 % | Width | n_eff |
|---|---|---|---|---|---|
| Naive binomial at *n* | 39.6 | 39.5 | 38.9 | 21.4 pts | 69 |
| HAC (Bartlett, automatic bandwidth) | 66.6 | 66.6 | 66.1 | 33.8 pts | 21 |
| Moving-block bootstrap | 63.1 | 63.0 | 63.0 | 29.6 pts | — |
| **Run-structure** | **94.0** | **94.1** | **94.2** | 70.8 pts | **5.6** |

## Why the standard corrections fail

This was the useful part of the exercise. Both HAC and the block bootstrap are
the obvious tools, both are large improvements on the naive interval, and both
still under-cover by roughly thirty points.

They fail for the same reason: **they mis-estimate how far the dependence
runs.** The conventional automatic Bartlett bandwidth is `4(n/100)^(2/9)`, which
evaluates to about **three bars** at these sample sizes. The dependence in a
real forecast stream runs to **forty**. Truncating the kernel at three discards
most of the covariance the formula is trying to sum.

A first version of this analysis used the automatic bandwidth, reported 57 %
coverage, and would have shipped a correction that does not correct. The failure
was only visible because coverage was measured rather than assumed — which is
the same discipline that produced every other finding here.

## What the corrected number says

`n_eff ≈ 6` is the practical content. An honest interval on a session hit-rate
is roughly **±35 points wide**, spanning everything from incompetence to
apparent mastery. That is the correct description of what a single session of a
per-minute forecast stream can establish about skill: essentially nothing.

The correction changes no point estimate. Only the interval moves. What it buys
is an interval that means what it claims, and the discipline of reporting
`n_eff` alongside `n` whenever forecasts overlap.

## Practical rule

When reporting a hit-rate from overlapping forecasts:

1. Compute the runs of constant hit indicator.
2. Report `n_eff = n / (Σ L_i² / n)` beside the raw `n`.
3. Build the interval at `n_eff`.
4. If `n_eff` is in single digits, say so, and do not draw a conclusion.

See [finding 02](02-measurement-illusions.md) for the artefact this corrects,
and [finding 10](10-methodology.md) for where it sits in the wider protocol.
