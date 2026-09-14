# 03 — Cost geometry

**Claim tested.** That a directional call above 50% is worth trading.

**Verdict.** On NSE index intraday, it is not — because the size of the typical
move is approximately the size of the round trip. Accuracy and profitability are
not the same variable, and at these horizons they point in opposite directions.

---

## The core arithmetic

| | BANKNIFTY | NIFTY 50 |
|---|---|---|
| Typical 30-minute move | ~36 pts | ~14 pts |
| Round-trip cost + slippage | ~27 pts | ~10 pts |
| **Cost as share of the move** | **~75%** | **~71%** |

Roughly 65–70% of the average move is consumed before the trade has any chance
to be right. What is left has to cover the losing half of a near-coinflip.

The consequence is measurable directly. Across all timeframes, the share of
minutes where the subsequent move is large enough to clear costs at all:

| Horizon | Share of bars where move > cost |
|---|---|
| 5m / 15m / 30m | 21–35% |
| 60m | **42%** (best) |

So on the majority of bars there is no profitable trade available in either
direction, regardless of how accurate the forecast is.

> Status: **RECORDED** (29 Jul 2026).

## What this does to an accurate system

A cost-aware backtest was run on the most selective configuration the engine
could produce: STRONG signal (3 of 4 horizons in the HIGH confidence tier and
agreeing) **plus** prime-hour window **plus** full indicator alignment, with
adaptive ATR stops, slippage, and one entry per setup.

Single session, 28 July 2026, scored as distinct trades:

| | BANKNIFTY | NIFTY 50 |
|---|---|---|
| Distinct trades | 8 | 9 |
| Win rate | 25% | 44% |
| Profit factor | 0.20 | 0.52 |
| **Net after cost + slippage** | **−₹4,629** | **−₹3,300** |

Both negative. The selectivity filters worked — they reduced trade count
sharply — but there was nothing on the other side of the filter to harvest.

> Status: **RECORDED** (28 Jul 2026).

## The result that made the point unavoidable

A meta-label filter was built on the López de Prado pattern: the primary signal
picks the side, a second model decides whether the setup is worth taking at all.
Crucially, it was trained on the question that actually matters —
*"will this trade earn at least 5 points per minute?"* — rather than
*"will the target be hit?"*.

The label choice matters enormously. Training on target-hit pushes the model
toward small, easy targets: the hit-rate climbs from 42% to 53% while the money
gets worse. Labelling on points-per-minute keeps the geometry honest.

Out-of-sample over 106 unseen days, BANKNIFTY, applied on top of the existing
speed and thrust gates:

| Configuration | Trades/day | Points per minute | Total |
|---|---|---|---|
| Gates only | 2.3 | +1.44 | **−3,777** |
| Gates + meta filter (top 30%) | 1.0 | +1.82 | **−981** |

The filter did exactly what it was designed to do. It raised per-minute quality
by 26% and cut trade count by more than half, and it held in both halves of the
test period. The total still lost money — it lost *less*.

That is the cleanest statement of the finding available: a working filter on a
system with no edge converts a large loss into a small one. It does not convert
it into a profit, because there is no profit in the underlying signal for the
filter to concentrate.

The same model failed out-of-sample on NIFTY 50 (+0.25 → +0.08 points/minute,
second half negative) and is therefore not applied there.

> Status: **REPRODUCIBLE** — `oracle/models/meta_filter.py`.

## The identity that replaces accuracy

```
expectancy  =  win_rate × avg_win  −  (1 − win_rate) × avg_loss
```

A 40% win rate at 1:2.5 reward-to-risk beats a 60% win rate at 1:0.8. In every
directional test in this project the payoff ratio came out near 1.0 — average
win ≈ average loss — which makes the win rate the only remaining lever, and it
sits at 50%.

Measured after a realistic 0.04% round-trip cost, expectancy at the 30-minute
horizon:

| | Expectancy per trade |
|---|---|
| BANKNIFTY | **−20 pts** |
| NIFTY 50 | **−8.5 pts** |

Negative at a ~50% hit-rate, exactly as the identity predicts.

## Why this points at options

The way out of the identity is not a better win rate. It is a payoff structure
that is not symmetric. That is what an option is: a position whose profit and
loss are shaped differently by construction.

This is the reasoning that leads to [finding 04](04-volatility-predictability.md)
(forecast the size of the move, which options pay for) and
[finding 05](05-variance-risk-premium.md) (sell the size of the move, which is
systematically overpriced).
