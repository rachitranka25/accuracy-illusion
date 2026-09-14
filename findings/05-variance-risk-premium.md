# 05 — The variance risk premium

**Claim tested.** That there exists at least one edge on this data that is both
measurable and reachable from a retail account.

**Verdict.** Exactly one: **selling NIFTY at-the-money premium.** +22.5 points
per trade after cost, 61% win rate, on real historical option premiums. It is
also the finding that requires the most careful handling, because its risk
profile is not symmetric with its returns.

---

## Result

Short at-the-money straddle, held to expiry, priced on real EOD option premiums
from the F&O bhavcopy (`data_cache/{name}_options_eod.csv`, 2 years).

| | NIFTY 50 | BANKNIFTY |
|---|---|---|
| Trades | 349–509 | 349–509 |
| Win rate | **61%** | lower |
| Net per trade after 3% cost | **+22.5 pts** | **−7.3 pts** |
| Worst single trade | −847 | **−2,002** |

> Status: **REPRODUCIBLE** — `research/archive/vrp_straddle_test.py`.

## Why it works

Implied volatility is persistently priced above subsequently realised
volatility. Option sellers are paid for carrying risk that buyers want to shed,
and that spread is the variance risk premium — one of the most durable
documented anomalies in derivatives markets, and structural rather than
informational. It does not depend on predicting anything.

That is precisely why it survived a test protocol that killed every predictive
signal in this project. It is not a prediction.

## Why BANKNIFTY does not work

Same logic, opposite outcome. BANKNIFTY's return distribution has a fatter tail —
its worst trade is −2,002 against NIFTY's −847. The premium collected is not
enough to pay for tails that size. The edge is index-specific, and assuming it
generalises would be the standard error.

## The counter-intuitive result

The obvious improvement is to sell premium only on days the volatility model
([finding 04](04-volatility-predictability.md)) predicts QUIET. It was tested.
**It makes profitability worse:**

| Filter | Net per trade |
|---|---|
| Unfiltered | +22.5 pts |
| Only the quietest 33% of predicted days | **+2.4 pts** |

The reason is that premium is richest exactly when the model predicts a big
move, because high predicted volatility means high implied volatility means a
larger credit. Filtering to quiet days filters out the paid days.

What the filter *does* buy is tail protection:

| | Unfiltered | Quiet-filtered |
|---|---|---|
| Worst trade | −847 | **−538** |
| Max drawdown | −5,405 | **−2,605** |

So the volatility model's correct role here is a **risk dial, not alpha**.
Turning it on halves the tail and halves the return. That is a position-sizing
decision, not a signal.

## The shape of the risk

A 61% win rate on small regular credits against occasional large debits is the
classic short-volatility profile. Stated plainly:

- The strategy makes money most weeks.
- A single bad week can erase months of it.
- A high win rate here is a **warning about hidden tail risk**, not evidence of
  skill.

Implications, none of them optional:

1. **Never naked.** Defined-risk structures only — iron condor, credit spread.
   `oracle/strategy/options.py` builds condors for this reason.
2. **Size for the tail, not the average.** The worst historical trade, not the
   mean, sets position size.
3. **The backtest holds to expiry.** Real positions face margin expansion on the
   path, which the backtest does not model. Margin is a real constraint that can
   force an exit at the worst moment.
4. **Costs were assumed at 3%.** Retail option spreads on non-index strikes can
   be wider. The edge is +22.5 points; the cost assumption is load-bearing.

## Status in the project

This is documented as the one measurable, reachable edge found. It is **not**
implemented as an automated strategy and it has **not** been traded. The gap
between "positive in a hold-to-expiry EOD backtest" and "survivable in a live
account with margin, spreads and path risk" is exactly where short-volatility
strategies historically fail.

The next honest step is a defined-risk condor advisor validated on real premiums
with margin modelling — not an execution engine.
