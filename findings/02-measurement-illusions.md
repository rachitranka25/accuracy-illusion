# 02 — Measurement illusions

**Claim tested.** That the high accuracy numbers this project repeatedly
produced — 71%, 85%, 62% — were evidence of an edge.

**Verdict.** Every one of them was an artefact of how the number was counted.
None survived being recounted correctly.

This is the finding the paper is named after, because it explains not only our
own false discoveries but the shape of a great deal of what gets published and
marketed in this field.

---

## The mechanism

Two facts combine, and **neither is sufficient on its own**. Getting this wrong
was itself instructive: a first version of the simulation modelled the forecaster
as re-rolling every bar, found no inflation at all, and had to be corrected.

**(a) Overlapping outcomes.** A *k*-bar-ahead forecast emitted every bar shares
*k−1* of its *k* outcome bars with its neighbour. Measured on real data, the
forward-return autocorrelation is 0.84 at lag 1 and decays to zero at exactly
lag *k* — the signature of arithmetic overlap, not of structure
([finding 01](01-directional-ceiling.md)).

**(b) Persistent predictions.** A real model does not re-roll every bar. Its
features move slowly, so it repeats the same call for long stretches. Measured
directly on this project's own live forecast log:

| Index | Forecasts | Distinct runs | Mean run | Longest run |
|---|---|---|---|---|
| BANKNIFTY (30m) | 1,500 | 99 | **15.2 min** | 135 min |
| NIFTY 50 (30m) | 1,500 | 123 | **12.2 min** | 207 min |

A model that genuinely re-decided every minute would show a mean run of 1.

When both the call and the outcome persist, one lucky trending stretch is
recorded as dozens of hits. With independent predictions the hit indicator stays
i.i.d. and overlap changes nothing — it is the combination that does the damage.

## What that does to a forecaster with zero skill

A fair coin held for the empirically measured run length, evaluated against real
BANKNIFTY and NIFTY 50 bars, scored the way a live dashboard scores it.

> Status: **REPRODUCIBLE** — `research/overlap_simulation.py`, 200 random
> forecasters per real session, ~70,000 simulated sessions per index.

| | Scored **every bar** | Scored **once per k bars** |
|---|---|---|
| Reported sample size per session | 69 | 12 |
| Stated binomial 95% CI | ±11.8 pts | ±28.3 pts |
| **Realised spread across sessions** | **±19.6 pts** | ±28.0 pts |
| **Understatement factor** | **1.66×** | **0.99×** |

The right-hand column is the control: scored correctly, the stated interval
matches the realised spread almost exactly. Scored every bar, the stated interval
is **1.66× too narrow**, because the reported *n* of 69 is a fiction — the
effective sample size is closer to 12.

The tail is where the damage shows:

| Session hit-rate | Probability (zero skill, scored every bar) |
|---|---|
| ≥ 60% | **15.9%** — 1 session in 6 |
| ≥ 71% | **1.97%** — 1 session in 51 |
| ≥ 80% | 0.06% |
| ≥ 85% | 0.01% |

## At the granularity it is actually reported

Everything above is on 5-minute bars, where a 30-minute horizon is k=6 and a
session holds 69 forecasts. The live system emits a forecast **every minute**, so
k=30 and a session holds ~314 — and the 202/285 that started this project was
counted at that density. Re-run there, with the outcome series taken from the
production log rather than simulated:

| | per 5-min bar | **per minute** |
|---|---|---|
| Forecasts per session | 69 | **314** |
| Effective sample | 12.4 | 16.5 |
| Stated s.d. | 6.02% | 2.82% |
| Realised s.d. | 10.00% | 8.76% |
| **Understatement factor** | **1.66×** | **3.11×** |
| P(session ≥ 71%) | 2.0% | 0.6–1.3% |

Two things, and they agree with the reframing above. The interval is **worse**
calibrated at the real granularity — 3.11× too narrow — because the nominal
count rises fivefold while the information does not. But extreme sessions become
slightly **rarer**, not commoner. Per-minute scoring does not manufacture more
71% readings; it manufactures more confidence in them.

Every 5-minute-bar figure in this record therefore **understates** the artefact.

> Status: **REPRODUCIBLE** — `research/per_minute.py`.

## Multiplicity: why 71% arrives within a fortnight

A dashboard does not watch one number. This one displays **2 indices × 4
horizons = 8 hit-rates every session**.

```
P(at least one series ≥ 71% on a given day) = 1 − (1 − 0.0197)^8 = 14.7%
                                            ≈ once every 7 sessions
```

So a headline 71% arrives in the first fortnight of live operation with **no
edge whatsoever**. And this project's own 71% was exactly that: one index, one
horizon, one trending Friday.

## Case study: the 71%

Traced to source, it was **NIFTY 50, 30-minute horizon, on one day** — a
trending Friday in July 2026.

> **The precise figures are withdrawn.** This document previously quoted
> "202 of 285 forecasts", 56% the following Monday and 51% on re-scoring. The
> portion of the log holding that session was lost before the current analysis,
> so none of it can be reproduced. A record that demands reproducibility cannot
> keep numbers it cannot produce.

The surviving log holds a close analogue, fully analysed in
[finding 12](12-corrected-estimator.md): a NIFTY 50 session reporting **69.6%
over 335 per-minute forecasts**, whose naive interval excludes 50% comfortably
and whose corrected interval [50.0, 89.1] barely does.

A head-to-head logger was built to settle the engine comparison live. Over a
full session the apparent morning lead reversed completely by the close — noise
in both directions, as the simulation above predicts.

## Case study: gap-fade

Fading the opening gap screened at **54–63%** per 5-minute bar across 17 months —
stable, high sample count, exactly what a real signal is supposed to look like.

But a gap happens **once per day**:

| Scoring unit | Accuracy |
|---|---|
| Per 5-minute bar | 54–63% |
| **Per day** | **44–48%** |

Below a coinflip when counted once.

## Case study: previous-day range breakout

This one passed every filter that had killed the others. NIFTY 50, 60-minute
horizon: **60% directional accuracy, stable across both halves, 291 distinct
events.** By the standards used up to that point, a discovery.

Run through a cost-aware backtest — 60-minute hold, ₹10 per lot round-trip, lot
size 75:

| | Result |
|---|---|
| Win rate | 51% |
| Net P&L | **−₹26,861** |
| Profit factor | **0.95** |
| BANKNIFTY equivalent | −₹60,000 |

A 60% directional call became a 51% *profitable* call once costs applied, and
51% at a symmetric payoff loses. This is the bridge to
[finding 03](03-cost-geometry.md).

## Case study: the 85% claim

A claim reached the project second-hand: 85% accuracy sustained over 76 trading
days. Three independent reasons it carries no information:

1. **It is the shape of the artefact above**, at the extreme tail — and our own
   screens produced 85% on a Wednesday morning and 42% on a Monday from the same
   model.
2. **Even a genuine edge can live and die inside 76 days** — see
   [finding 07](07-edge-decay.md), where a real, cost-positive signal went from
   74% to 38% in under two years.
3. **Sustained 85% is not the tail of the distribution we measured.** At 0.01%
   per session, it is not reachable by overlap alone across 76 days. Which means
   either a shorter effective sample than reported (fewer forecasts, a partial
   session), a target so small it does not clear costs
   ([finding 03](03-cost-geometry.md)), or lookahead. Those are the options.

## The rules that came out of this

Now applied to every result in this repository:

1. **Score one observation per event**, never per bar, when the event is not
   per-bar.
2. **Collapse overlapping signals** into distinct trades before computing a
   hit-rate.
3. **State the effective sample size**, not the raw count, and widen the
   interval to match.
4. **Correct for multiplicity** — a grid of indices × horizons is a family of
   tests ([finding 11](11-multiple-testing.md)).
5. **A hit-rate without a payoff ratio is not a result.**
6. **Single-day live numbers are illustrations, not evidence.**
7. **Treat a surprisingly high number as a bug report** about the measurement
   until the measurement has been re-derived.

Rule 7 is the one that matters. Every high number in this project was a bug in
the measurement, and looking for the bug was always faster than celebrating.
