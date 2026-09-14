# 03 — Cost geometry: the accuracy that would actually be required

**Claim tested.** That a directional call above 50% is worth trading.

**Verdict.** On NSE index intraday it is not, and the gap is not marginal.
At the 30-minute horizon BANKNIFTY requires **71.5%** accuracy to break even on
futures costs. The best model family in
[finding 01](01-directional-ceiling.md) achieved **51.3%**. At option costs the
requirement exceeds 100% — meaning **no directional accuracy whatsoever is
sufficient**.

> Status: **REPRODUCIBLE** — `research/cost_geometry.py`,
> [`results/cost_geometry.json`](../results/cost_geometry.json).

---

## The arithmetic

For a symmetric bracket — target and stop both set at the typical move *m*,
round-trip cost *c* — expectancy per trade is

```
E = p(m − c) − (1 − p)(m + c) = m(2p − 1) − c
```

so break-even requires

```
p* = ½ + c / (2m)
```

The ratio **c/m is the entire game.** Where it is small, a 52% edge is a
business. Where it approaches 1, no achievable accuracy is enough.

*(The symmetric bracket is a modelling choice. Widening the target relative to
the stop lowers the required hit-rate but lowers the achieved one too; across
every configuration tested in this project the payoff ratio came out near 1.0,
which is why the symmetric case is the honest reference.)*

## Measured, at 4 bp round-trip cost

4 bp is a realistic futures figure: brokerage, STT, exchange charges, stamp duty
and slippage.

### BANKNIFTY

| Horizon | Median move | c/m | **Break-even accuracy** | Bars where move > cost | Achieved |
|---|---|---|---|---|---|
| 5m | 22.1 pts | 1.01 | **100.7%** | 49% | — |
| 15m | 37.7 pts | 0.60 | **79.8%** | 68% | — |
| 30m | 52.3 pts | 0.43 | **71.5%** | 76% | **51.3%** |
| 60m | 72.9 pts | 0.31 | **65.4%** | 82% | — |

### NIFTY 50

| Horizon | Median move | c/m | **Break-even accuracy** | Bars where move > cost | Achieved |
|---|---|---|---|---|---|
| 5m | 8.1 pts | 1.22 | **111.2%** | 42% | — |
| 15m | 14.1 pts | 0.70 | **84.9%** | 63% | — |
| 30m | 19.7 pts | 0.50 | **75.1%** | 73% | **50.1%** |
| 60m | 27.7 pts | 0.35 | **67.7%** | 80% | — |

Two things fall out immediately.

**At the 5-minute horizon the requirement is above 100% on both indices.** The
median 5-minute move is smaller than the round trip. There is no accuracy — not
90%, not 100% — at which scalping these indices on 5-minute signals is
profitable at retail cost. The shorter the horizon, the more certainly it fails,
which is the exact opposite of how short-horizon strategies are marketed.

**Shorter horizons are worse, and that is structural.** Move size grows roughly
with the square root of time while cost is fixed per round trip, so c/m falls
monotonically as the horizon extends. Every incentive in the data points away
from frequent trading.

## Sensitivity: what the instrument does to the requirement

BANKNIFTY, 30-minute horizon, median move 9.3 bp:

| Round-trip cost | Instrument | Break-even accuracy |
|---|---|---|
| 1 bp | exchange fees only, no slippage | 55.4% |
| 2 bp | tight futures, minimal slippage | 60.8% |
| 4 bp | realistic futures round trip | **71.5%** |
| 8 bp | futures with adverse slippage | 93.0% |
| 20 bp | **ATM options: spread + 30 min of theta** | **impossible** |
| 40 bp | OTM options or a wide book | **impossible** |

NIFTY 50 is worse still — it becomes impossible at 8 bp.

"Impossible" is not rhetoric. It is the case where c > m, so p* > 1: the cost of
the round trip exceeds the typical move, and a perfect forecaster still loses on
the median trade.

**This is the single most important row in this repository for a retail options
trader.** The instrument most retail participants actually use, over the horizon
they actually hold, has no directional accuracy that makes it profitable.

## Statistical significance is not economic significance

The model zoo found a small but genuinely significant effect on BANKNIFTY:
XGBoost at 51.3%, p = 0.001, against a shuffled-label control at 49.7%. That is a
real 1.6-point edge over noise at n = 16,856, and it survives Bonferroni
correction across all 18 model-index tests run.

It is also worth nothing. 51.3% against a 71.5% requirement loses on every
trade. The edge is real and the strategy is not.

This distinction is the reason accuracy is the wrong headline metric, and the
reason this project's fourteen months of accuracy-chasing produced no tradeable
result even where it produced a statistically defensible one.

## What this does to the "85%" claims

Take the claim at face value for a moment. Suppose someone genuinely achieves
85% directional accuracy at the 30-minute horizon.

| | BANKNIFTY | NIFTY 50 |
|---|---|---|
| Break-even requirement | 71.5% | 75.1% |
| A genuine 85% clears it by | **+13 pts** | **+10 pts** |

So even a real 85% — a figure well above anything in the peer-reviewed
literature — is a modestly profitable futures strategy, not a fortune. On
options it remains unprofitable.

A claim of 85% is therefore either a measurement artefact
([finding 02](02-measurement-illusions.md)), or true and far less valuable than
it sounds. Both readings argue against paying for it.

## Corroboration from the live system

A cost-aware backtest of the most selective configuration the engine could
produce — STRONG signal (3 of 4 horizons in the HIGH tier and agreeing), prime
window, full indicator alignment, adaptive ATR stops, slippage, one entry per
setup — scored as distinct trades on 28 July 2026:

| | BANKNIFTY | NIFTY 50 |
|---|---|---|
| Distinct trades | 8 | 9 |
| Win rate | 25% | 44% |
| Profit factor | 0.20 | 0.52 |
| Net after cost + slippage | **−₹4,629** | **−₹3,300** |

> Status: **RECORDED** (28 Jul 2026).

## The filter that worked and still lost

A meta-label filter on the López de Prado pattern: the primary signal picks the
side, a second model decides whether the setup is worth taking. Trained on
*"will this trade earn at least 5 points per minute?"* rather than
*"will the target be hit?"* — the label choice matters enormously, since
training on target-hit pushes the model toward small easy targets and raises the
hit-rate from 42% to 53% while making the money worse.

Out-of-sample over 106 unseen days, BANKNIFTY:

| Configuration | Trades/day | Points per minute | Total |
|---|---|---|---|
| Gates only | 2.3 | +1.44 | **−3,777** |
| Gates + meta filter (top 30%) | 1.0 | +1.82 | **−981** |

The filter did what it was designed to do — 26% better per-minute quality, less
than half the trades, and it held in both halves of the test. The total still
lost money.

**A working filter on a system with no edge converts a large loss into a small
one.** It cannot manufacture a profit that is not in the underlying signal.

The same model failed out-of-sample on NIFTY 50 (+0.25 → +0.08 points/minute,
second half negative) and is not applied there.

> Status: **REPRODUCIBLE** — `oracle/models/meta_filter.py`.

## Why this points at options — as structures, not as direction

The way out is not a better win rate against a symmetric payoff. It is a payoff
that is not symmetric by construction. That is what an option is.

This leads to [finding 04](04-volatility-predictability.md) — forecast the
*size* of the move, which options are priced on — and
[finding 05](05-variance-risk-premium.md) — sell the size of the move, which is
systematically overpriced. Both sidestep the arithmetic above instead of trying
to beat it.
