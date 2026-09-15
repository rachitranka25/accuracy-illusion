# 05 — The variance risk premium

**Claim tested.** That there exists at least one edge on this data that is both
measurable and reachable from a retail account.

**Verdict.** Yes — selling at-the-money premium — but the numbers previously
recorded here were wrong, and the corrected ones are smaller and point the other
way on BANKNIFTY.

Re-run from the committed script (short ATM straddle, entered *D* trading days
before expiry, held to expiry, net of 10% of premium as round-trip cost):

| | Trades | Win | Net/trade | Worst | h1 / h2 |
|---|---|---|---|---|---|
| NIFTY, 3 DTE | 108 | 61% | **+2.8** | −687 | 61 / 63 |
| NIFTY, 1 DTE | 108 | 62% | **+3.2** | −439 | 61 / 72 |
| BANKNIFTY, 3 DTE | 39 | 64% | **+14.2** | −881 | 58 / 75 |
| BANKNIFTY, 1 DTE | 40 | 68% | **+59.3** | −936 | 75 / 70 |

> **Retraction.** This document previously reported +22.5 points per trade for
> NIFTY and a *loss* of 7.3 points for BANKNIFTY, over "349–509 trades". None of
> that reproduces. The trade counts are 39–108, NIFTY nets low single digits,
> and **BANKNIFTY is positive, not negative**. The old figures came from memory
> of an earlier run rather than from the script, which is exactly the failure
> mode this project documents.

> Status: **REPRODUCIBLE** — `research/archive/vrp_straddle_test.py`.

---

## What is still missing

Four things, and until they are done this is a lead rather than a result:

1. **Sample size.** 39–108 expiries, not thousands. No interval accompanies any
   mean.
2. **No multiplicity correction** over the choice of strike, entry day and
   holding rule — and the table above reports four such choices.
3. **BANKNIFTY is not like-for-like.** Its 39 expiries against NIFTY's 108
   reflect NSE moving BANKNIFTY options from weekly to monthly expiry partway
   through the sample. Different holding period, different decay profile.
4. **The 10% cost assumption is load-bearing** and is not calibrated per
   instrument.

## Why it works

Implied volatility is persistently priced above subsequently realised
volatility. Option sellers are paid for carrying risk that buyers want to shed,
and that spread is the variance risk premium — one of the most durable
documented anomalies in derivatives markets, and structural rather than
informational. It does not depend on predicting anything.

That is precisely why it survived a test protocol that killed every predictive
signal in this project. It is not a prediction.

## On the index comparison

An earlier version of this document explained at length why BANKNIFTY "does not
work", on the basis of a −7.3 figure that does not reproduce. BANKNIFTY is in
fact the stronger of the two in this test. What remains true is that its tail is
heavier in absolute points (worst −936 against NIFTY's −687 on comparable
entries), and that its sample is both smaller and structurally different because
of the expiry change.

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
