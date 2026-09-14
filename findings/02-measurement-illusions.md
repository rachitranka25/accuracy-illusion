# 02 — Measurement illusions

**Claim tested.** That the high accuracy numbers this project repeatedly
produced — 71%, 85%, 62% — were evidence of an edge.

**Verdict.** Every one of them was an artefact of how the number was counted.
None survived being recounted correctly.

This is the most useful finding in the project, because it is the one that
explains why so many published and marketed strategies look profitable.

---

## The overlapping-sample inflation

The live engine emits a forecast **every minute** over a 30-minute horizon.
Consecutive forecasts therefore share 29 of their 30 minutes of outcome. They
are not independent observations — they are one observation sampled thirty times.

If the market trends for half an hour and the model said UP, that single correct
call is recorded as roughly thirty hits. If it trends the other way, one wrong
call becomes thirty misses. The result is a hit-rate that swings violently
between sessions and averages out to something far higher than the tradeable
reality on any trending day.

Measured directly, by collapsing consecutive identical signals into one distinct
trade:

| Scoring method | BANKNIFTY | NIFTY 50 |
|---|---|---|
| Per-minute forecasts | 48–68% | 48–68% |
| **Distinct trades** | **26%** | **66%** (n = 9) |

The inflation factor was **8–11×** in the number of "observations". The Nifty
number moves the other way purely because nine trades is not a sample.

> Status: **RECORDED** (28 Jul 2026, A+ selective backtest).

## Case study: the 71%

An early screen reported 71% directional accuracy and it became the project's
reference point for months. Traced to source, it was:

- **NIFTY 50, 30-minute horizon, on one day** — Friday 24 July 2026 — where
  202 of 285 per-minute forecasts were scored as hits.
- The *same* configuration scored **56%** the following Monday.
- Reproduced properly (train strictly before 24 July, score 24 July out-of-sample,
  one observation per 5-minute bar rather than per minute): **51%**.
- The older "legacy" engine that was believed to be responsible: **48%** on the
  same correct scoring.

So 71% was a real number on a screen, produced by counting a single trending
Friday thirty times. It was never an edge, and neither engine was better than
the other.

> Status: **RECORDED** (27 Jul 2026). A head-to-head logger was built to settle
> the engine comparison live; over a full session the apparent lead reversed
> completely by the close — new-vs-legacy differences were noise in both
> directions.

## Case study: gap-fade

Fading the opening gap screened at **54–63%** per 5-minute bar across 17 months —
stable, high sample count, exactly what a real signal is supposed to look like.

But a gap happens **once per day**. Scored as one observation per day, which is
the only honest unit for a once-daily event:

| Scoring unit | Accuracy |
|---|---|
| Per 5-minute bar | 54–63% |
| **Per day** | **44–48%** |

The per-bar version was counting the same daily event dozens of times. Below a
coinflip when counted once.

> Status: **RECORDED** (29 Jul 2026).

## Case study: previous-day range breakout

This one passed every filter that had killed the others. NIFTY 50, 60-minute
horizon: **60% directional accuracy, stable across both halves of the sample,
291 distinct events.** By the standards used up to that point, it was a
discovery.

It was then run through a cost-aware backtest — 60-minute hold, ₹10 per lot
round-trip cost, lot size 75:

| | Result |
|---|---|
| Win rate | 51% |
| Net P&L | **−₹26,861** |
| Profit factor | **0.95** |
| BANKNIFTY equivalent | −₹60,000 |

A 60% directional call became a 51% *profitable* call once the cost threshold
was applied, and 51% at a symmetric payoff loses. This is the bridge to
[finding 03](03-cost-geometry.md).

> Status: **RECORDED** (29 Jul 2026).

## Case study: the 85% claim

A claim reached the project second-hand: 85% accuracy sustained over 76 trading
days. Two independent reasons it carries no information:

1. **It is the shape of the illusion above.** Our own screens produced 85% on a
   Wednesday morning and 42% on a Monday from the same model. Per-minute
   counting on trending days reproduces exactly this number without any edge.
2. **Even a genuine edge can live and die inside 76 days.** See
   [finding 07](07-edge-decay.md), where a real, cost-positive signal went from
   74% to 38% in under two years.

## The rules that came out of this

These are now applied to every result in this repository:

1. **Score one observation per event**, never per bar, when the event is not
   per-bar. Daily events get daily scoring.
2. **Collapse overlapping signals** into distinct trades before computing a
   hit-rate.
3. **A hit-rate without a payoff ratio is not a result.** Report expectancy.
4. **Single-day live numbers are illustrations, not evidence.**
5. **Treat a surprisingly high number as a bug report** about the measurement
   until the measurement has been re-derived.

Rule 5 is the one that matters. Every high number in this project was a bug in
the measurement, and looking for the bug was always faster than celebrating.
