# 05 — The variance risk premium: withdrawn as an edge

**Claim tested.** That selling at-the-money premium is the one measurable,
retail-reachable edge this project found.

**Verdict.** **Withdrawn.** It does not survive its own multiplicity correction.
No entry configuration is significantly profitable on either index, every
bootstrap interval on the mean contains zero, and the best of four
configurations is *worse* than what a recentred null typically produces.

> Status: **REPRODUCIBLE** — `research/vrp_test.py`,
> [`results/vrp_test.json`](../results/vrp_test.json).

---

## The numbers, with dispersion this time

Short ATM straddle held to expiry, cost 10% of premium collected. Every entry
point tried is shown, not the best one.

### NIFTY (weekly expiries, median gap 7 days)

| DTE | n | Win | Net/trade | s.d. | t | p | 95% CI |
|---|---|---|---|---|---|---|---|
| 1 | 108 | 62% | 3.2 | 123 | 0.27 | 0.394 | [−21, 26] |
| 2 | 108 | 62% | 12.1 | 168 | 0.75 | 0.228 | [−20, 43] |
| 3 | 108 | 61% | 2.8 | 214 | 0.14 | 0.446 | [−39, 43] |
| 5 | 107 | 48% | **−45.5** | 254 | −1.85 | 0.967 | [−93, 3] |

### BANKNIFTY (monthly expiries, median gap 27 days)

| DTE | n | Win | Net/trade | s.d. | t | p | 95% CI |
|---|---|---|---|---|---|---|---|
| 1 | 40 | 68% | 59.3 | 315 | 1.19 | 0.121 | [−43, 149] |
| 2 | 40 | 65% | 58.1 | 553 | 0.66 | 0.255 | [−117, 214] |
| 3 | 39 | 64% | 14.1 | 472 | 0.19 | 0.426 | [−124, 153] |
| 5 | 39 | 49% | **−209.2** | 624 | −2.09 | 0.979 | [−408, −15] |

**0 of 4 significantly profitable** on either index, raw or after
Benjamini–Hochberg.

## The null beats the winner

Recentre each configuration's P&L to a zero mean, resample, and record the best
of four. That is the distribution of "best backtest of N" when there is no edge:

| | Observed best | Null best (median) | Null best (95th) | P(null ≥ obs) |
|---|---|---|---|---|
| NIFTY | 12.1 | **17.8** | 43.0 | **0.679** |
| BANKNIFTY | 59.3 | **75.1** | 181.8 | **0.620** |

Same pattern as the 86-strategy sweep in [finding 11](11-multiple-testing.md):
selecting the best of a handful of choices produces a better-looking result than
the one we actually found.

## The two indices were never comparable

NSE moved BANKNIFTY options from weekly to monthly expiry partway through the
sample. Split on expiry spacing (3 DTE entry):

| BANKNIFTY | n | Net/trade |
|---|---|---|
| Weekly sub-sample | 17 | **−67.5** |
| Monthly sub-sample | 21 | **+57.5** |

The positive headline is entirely the monthly regime, on 21 expiries. NIFTY has
no monthly sub-sample to compare against. An external auditor flagged this
before we tested it.

## The cost assumption sets the sign

| Cost (% of premium) | NIFTY | BANKNIFTY |
|---|---|---|
| 5% | +17.0 | +50.9 |
| 10% | +2.8 | +14.1 |
| 20% | **−25.6** | **−59.4** |

The sign of the result is chosen by an assumption, not by the data.

## What survives

The variance risk premium itself is real and well documented (Carr & Wu 2009;
Bakshi & Kapadia 2003). Mean premium does exceed mean subsequent move in most
configurations here, and the 61–68% win rates are genuine.

What does not survive is **our measurement of it as an edge**. A 61% win rate on
a left-skewed payoff whose standard deviation is forty times its mean is not
evidence of profitability. It is the shape that makes short-volatility
strategies dangerous, and it is exactly the "high win rate as a warning" pattern
this record has flagged elsewhere while failing to apply it here.

## Retraction history

This document has now been corrected twice:

1. First it reported **+22.5 pts/trade (NIFTY)** and **−7.3 (BANKNIFTY)** over
   "349–509 trades". None of that reproduced; the real counts are 39–108 and
   BANKNIFTY was *positive*.
2. Now the corrected figures themselves fail multiplicity correction, so the
   **claim of an edge is withdrawn entirely.**

Both corrections came from running the code rather than re-reading the notes.
See [finding 10](10-methodology.md).
