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
of 15.2 minutes (3.0 bars) and a longest run of 205 minutes (41 bars).

This is the classical **design effect** (Kish; Moulton 1990) with clusters given
by the runs. Nothing about the statistic is new; what is new is noticing that a
forecast stream has exactly this structure and that nobody reports it.

## Validation by coverage

An interval is judged by whether it covers. A forecaster's calls persist per the
measured run-length distribution while the market does not, so the hit sequence
is only partly block-structured. Its true long-run hit-rate is measured rather
than assumed, and we count how often a nominal 95 % interval contains it.

BANKNIFTY, 30-minute horizon, nominal n = 69. NIFTY 50 agrees within 1 point.

| Estimator | 50 % | 55 % | 59 % | Width | n_eff |
|---|---|---|---|---|---|
| Naive binomial at *n* | 75.5 | 74.8 | 74.0 | 23.1 pts | 69 |
| HAC (Bartlett, automatic bandwidth) | 87.0 | 86.6 | 85.6 | — | 21 |
| Moving-block bootstrap | 86.6 | 86.4 | 85.2 | — | — |
| **Run-structure design effect** | **99.5** | **99.6** | **99.6** | 56.0 pts | **12.4** |

None is exactly calibrated. The naive interval under-covers by twenty points,
the two standard corrections by eight, and the design effect over-covers.

## Why each one misses

**HAC and the bootstrap under-cover** for the same reason, and it follows from
the formula. The conventional automatic Bartlett bandwidth `4(n/100)^(2/9)`
evaluates to **3.7 bars** at n = 69 — almost exactly the *mean* run of 3.0 bars.
But the inflation is driven by `E[L²]`, so it is the **tail** that matters, and
the tail reaches 41 bars. A kernel truncated near the mean has summed almost
none of the covariance the long runs contribute.

**The design effect over-covers** because it treats runs as independent
clusters. Consecutive runs are negatively dependent — a run of hits ends
precisely when a miss begins — so the formula overstates the dependence. It is
therefore a conservative bound, and should be reported as one rather than
tuned until it hits 95 %.

## What the corrected number says

`n_eff ≈ 12` against a nominal 69 is the practical content. An honest interval
on a session hit-rate is **±25 to ±28 points wide**, spanning everything from
incompetence to apparent mastery. That is the correct description of what one
session of a per-minute forecast stream can establish about skill: essentially
nothing.

The correction changes no point estimate. Only the interval moves.

## Practical rule

When reporting a hit-rate from overlapping forecasts:

1. Compute the runs of constant hit indicator.
2. Report `n_eff = n / (Σ L_i² / n)` beside the raw `n`.
3. Build the interval at `n_eff`, and treat the design effect as a conservative
   bound rather than an exact correction.
5. If `n_eff` is in low double digits or below, say so, and do not draw a
   conclusion from a single session.

See [finding 02](02-measurement-illusions.md) for the artefact this corrects,
and [finding 10](10-methodology.md) for where it sits in the wider protocol.
