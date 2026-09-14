# Findings — executive summary

Fourteen months of NSE index data, 100+ strategies, 7 model families, 3 data
regimes. What held, what did not, and what it cost to find out.

Full write-ups: [`findings/`](findings/). Method: [`findings/10-methodology.md`](findings/10-methodology.md).

---

## The headline

| Question | Answer | Evidence |
|---|---|---|
| Can intraday **direction** be predicted from free data? | **No.** ~50% out-of-sample, every model family, and shuffled labels score the same. | [01](findings/01-directional-ceiling.md) |
| Then why did our screens show 60–85%? | Overlapping samples inflate hit-rates **8–11×**. Every high number traced back to a counting error. | [02](findings/02-measurement-illusions.md) |
| Does a 53% call make money? | **No.** Cost + slippage consumes 65–70% of the typical move. | [03](findings/03-cost-geometry.md) |
| Is **magnitude** predictable? | **Yes — 68–74% out-of-sample, stable.** The strongest real result. | [04](findings/04-volatility-predictability.md) |
| Is anything actually profitable? | **One thing.** Short NIFTY ATM premium: +22.5 pts/trade after cost, 61% win. Fat left tail. | [05](findings/05-variance-risk-premium.md) |
| Does open interest know more than price? | **Marginally.** IC 0.04–0.06 over 2,181 days, stable — where price-only features show nothing. | [06](findings/06-open-interest.md) |
| Do edges last? | **No.** The best one went 74% → 38% in under two years. | [07](findings/07-edge-decay.md) |

## The three numbers that matter

**~50%** — directional accuracy out-of-sample, across Gradient Boosting,
LightGBM, XGBoost, LSTM, Transformer, TFT and kNN. An over-capacity model
memorises the training set to 96% and still scores 49% on test. **Shuffled
random labels score 50% on the same pipeline** — the strongest available
evidence that the signal is not there to be found.

**8–11×** — the inflation factor from scoring overlapping forecasts. A
30-minute forecast emitted every minute shares 29 of 30 minutes of outcome with
its neighbour. One good trending afternoon becomes thirty hits. Collapsed into
distinct trades, a 48–68% per-minute hit-rate became 26%.

**+22.5 points** — net per trade, after 3% costs, selling NIFTY at-the-money
straddles over 349–509 trades at a 61% win rate. The only positive expectancy
found. The same structure on BANKNIFTY loses −7.3 points per trade, because its
tail is fatter: worst trade −2,002 versus NIFTY's −847.

## What was retracted

A research log that edits its own history is worthless. Four documented positive
results were withdrawn after better tests:

| Claim | What it actually was |
|---|---|
| "71% accuracy" | One trending Friday counted per-minute. ~51% scored correctly. |
| "ORB is a positive-expectancy engine" | A lucky 60-day window. −18.6R over 351 sessions. |
| "VIX + cross-index features lift BANKNIFTY 2–4.5 pts" | Train/serve skew — features built from 2 years of history at training, today's bars at serve. |
| "PCR extremes are profitable" | True in 2024 (+₹100k, PF 1.11). 36% accuracy and PF 0.74 in the last six months. |

## What survived

1. **Volatility is forecastable.** 68% BANKNIFTY / 74% NIFTY 50 out-of-sample,
   stable across both halves, direction-free. Caveat: persistence already scores
   ~80% on BANKNIFTY, so the genuine *edge* is NIFTY 50 at 30–60 minutes
   (+9 to +13 points over baseline).

2. **The variance risk premium is real and reachable.** It is structural, not
   predictive, which is why it survived a protocol that killed everything
   predictive. It is also a short-volatility payoff: small regular gains,
   occasional large losses. Defined-risk structures only.

3. **Option-chain residuals carry a little information.** Over a decade,
   four features survive after the price component is regressed out — and the
   change in put-call ratio (flow) beats its level (position). Too small to
   trade alone; usable as context.

4. **One open lead.** A fixed pinning rule — fade the stretch from the OI
   cluster — returns 60% and +0.211% per trade over 55 out-of-sample BANKNIFTY
   days, which *would* clear costs. 55 trades is not a sample, and the decade
   test puts the same feature family at 52%. The correct next step is to run the
   identical fixed rule over the decade of chain data now on disk. Not yet done.

## What would be needed to go further

Not a better model. The ceiling test rules that out. Different information:

- **Order-flow imbalance** — the one short-horizon edge documented in
  peer-reviewed microstructure work. Historical data is paid-only, so a live
  collector runs instead. ~960 minutes accumulated. Untested.
- **Intraday open interest** — whether positions are being built or unwound
  *during* the session, which the price series does not carry. ~1,000 rows
  accumulated. Untested.
- **Real option premiums intraday** — to validate the volatility model's
  intended use, which is buying and selling structures rather than direction.

## Standing conclusion

Direction over short horizons is a coinflip that costs money to play. Magnitude
is genuinely forecastable but does not by itself constitute a trade. The only
edge both measurable and reachable is selling variance, and it pays in small
regular amounts while risking large irregular ones.

Nothing in this repository is trading advice, and nothing in it should be traded
with money that matters.
