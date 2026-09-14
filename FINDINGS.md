# Findings — executive summary

Fourteen months of NSE index data. Nine model families, 86 rule-based
strategies, three data regimes, and the statistics required to interpret that
many trials.

Full write-ups: [`findings/`](findings/) · Method:
[`findings/10-methodology.md`](findings/10-methodology.md) · Raw output:
[`results/`](results/) · Reproduction: [`research/`](research/).

---

## The headline

| Question | Answer | Evidence |
|---|---|---|
| Can intraday **direction** be predicted from free data? | A real edge of **~1 point** exists on BANKNIFTY and is statistically solid. It is economically worthless. | [01](findings/01-directional-ceiling.md) |
| Then why did our screens show 60–85%? | A **zero-skill** forecaster posts ≥71% once every 7 sessions on a dashboard watching 8 series. | [02](findings/02-measurement-illusions.md) |
| What accuracy would actually be needed? | **71.5%** on BANKNIFTY futures. At option costs, **no accuracy suffices**. | [03](findings/03-cost-geometry.md) |
| Is **magnitude** predictable? | **Yes — 68–74% out-of-sample, stable.** The strongest real result. | [04](findings/04-volatility-predictability.md) |
| Is anything actually profitable? | **One thing.** Short NIFTY ATM premium: +22.5 pts/trade after cost. Fat left tail. | [05](findings/05-variance-risk-premium.md) |
| Does open interest know more than price? | **Marginally.** IC 0.04–0.06 over 2,181 days, stable, where price-only shows nothing. | [06](findings/06-open-interest.md) |
| Do edges last? | **No.** The best went 74% → 38% in under two years. | [07](findings/07-edge-decay.md) |
| Is the best of 86 strategies a finding? | **No.** A block-bootstrap null produces a *better* winner 66% of the time. | [11](findings/11-multiple-testing.md) |
| Can the inflation be corrected? | **Yes.** A run-structure estimator covers at 94% where the standard one covers at 40%. | [12](findings/12-corrected-estimator.md) |

## The four numbers that matter

**51.3%** — the best out-of-sample directional accuracy from nine model
families on identical purged walk-forward splits (XGBoost, BANKNIFTY, 30-minute
horizon, n = 16,856, p = 0.001, shuffled-label control 49.7%). This is a genuine
effect that survives Bonferroni correction across all 18 model-index tests. It
does not replicate on NIFTY 50.

**71.5%** — the accuracy required to break even on the same instrument and
horizon at 4 bp round-trip cost. The gap between what is achievable and what is
required is **20 percentage points**. At ATM option costs over a 30-minute hold,
the round trip exceeds the median move and the requirement passes 100%: no
directional accuracy whatsoever is profitable.

**1 in 7** — how often a forecaster with exactly zero skill posts a ≥71%
session, on a dashboard displaying 2 indices × 4 horizons. Measured, not
assumed: a fair coin held for the engine's empirically observed run length
(15.2 minutes) against real bars. The stated confidence interval of a per-bar
hit-rate is **1.66× too narrow**; scored one observation per horizon it is
correctly calibrated at 0.99×.

**+22.5 points** — net per trade after 3% costs, selling NIFTY at-the-money
straddles over 349–509 trades at a 61% win rate. The only positive expectancy
found in the entire investigation. The same structure on BANKNIFTY loses 7.3
points per trade, because its tail is fatter: worst trade −2,002 versus NIFTY's
−847.

## The correction

Diagnosing a biased statistic is half a contribution. The other half is the fix.

The variance inflation has a closed form, and because the dependence is
block-shaped it collapses to `n_eff = n·E[L]/E[L²]` over the runs of constant
hit indicator — governed by the **second** moment of the run-length
distribution, which is why a heavy tail matters far more than a long mean.

Validated by coverage against a forecaster of known skill: the naive interval
covers 40% of the time while claiming 95%, a textbook HAC correction reaches
67%, a moving-block bootstrap 63%, and the run-structure estimator **94%**.

Both standard corrections fail for the same reason, and finding it was the point:
the conventional automatic Bartlett bandwidth is about three bars here, while
the dependence runs to forty.

The practical number is `n_eff ≈ 6` for a session that reports 69 forecasts. An
honest interval is ±35 points wide. One session establishes nothing.

## Statistical significance is not economic significance

This is the result the project turns on, and it took fourteen months to reach.

The models do find something. XGBoost's 1.6-point edge over its shuffled control
is real at n = 16,856. Every instinct trained on machine-learning benchmarks
reads p = 0.001 as success.

It loses money on every trade, because 51.3% against a 71.5% requirement is not
close. An effect can be simultaneously well-established and worthless, and in
this domain that is the normal case rather than the exception.

## What was retracted

A research log that edits its own history is worthless. Five documented positive
results were withdrawn after better tests:

| Claim | What it actually was |
|---|---|
| "71% accuracy" | One trending Friday counted per-minute. ~51% scored correctly, and reachable by a coinflip every 7 sessions. |
| "ORB is a positive-expectancy engine" | A lucky 60-day window. −18.6R over 351 sessions. |
| "VIX + cross-index features lift BANKNIFTY 2–4.5 pts" | Train/serve skew — features built from 2 years of history at training, today's bars at serve. |
| "PCR extremes are profitable" | True in 2024 (+₹100k, PF 1.11). 36% accuracy, PF 0.74 in the last six months. |
| "The strategy sweep found a stable 60% breakout" | 51% profitable after costs, −₹26,861, PF 0.95. |

## What survived

1. **Volatility is forecastable.** 68% BANKNIFTY / 74% NIFTY 50 out-of-sample,
   stable across both halves, direction-free. Caveat: persistence already scores
   ~80% on BANKNIFTY, so the genuine *edge* is NIFTY 50 at 30–60 minutes
   (+9 to +13 points over baseline).

2. **The variance risk premium is real and reachable.** Structural rather than
   predictive, which is why it survived a protocol that killed everything
   predictive. Also a short-volatility payoff: small regular gains, occasional
   large losses. Defined-risk structures only.

3. **Option-chain residuals carry a little information.** Over a decade, four
   features survive after the price component is regressed out — and the
   *change* in put-call ratio beats its *level*. Flow beats position. Too small
   to trade alone.

4. **One open lead.** A fixed pinning rule — fade the stretch from the OI
   cluster — returns 60% and +0.211% per trade over 55 out-of-sample BANKNIFTY
   days, which *would* clear costs. 55 trades is not a sample, and the decade
   test puts the same feature family at 52%. The correct next step is to run the
   identical fixed rule over the decade of chain data now on disk. Not yet done.

## What would be needed to go further

Not a better model. The ceiling test rules that out: an over-capacity model
memorises the training set to 100% and generalises at 50.5%, and scores 50.2% on
randomly shuffled labels. The pipeline recovers injected synthetic signal at
both 10% and 30% strength, so it can detect signal — there is none to detect.

Different information:

- **Order-flow imbalance** — the one short-horizon edge documented in
  peer-reviewed microstructure work. Historical data is paid-only, so a live
  collector runs instead. ~960 minutes accumulated. Untested.
- **Intraday open interest** — whether positions are being built or unwound
  *during* the session, which the price series does not carry. ~1,000 rows
  accumulated. Untested.
- **Real option premiums intraday** — to validate the volatility model's
  intended use, which is trading structures rather than direction.

## Standing conclusion

Direction over short horizons carries a real but tiny signal, an order of
magnitude too small to pay for the round trip. Magnitude is genuinely
forecastable but is not by itself a trade. The only edge both measurable and
reachable is selling variance, and it pays in small regular amounts while
risking large irregular ones.

Nothing in this repository is trading advice, and nothing in it should be traded
with money that matters.
