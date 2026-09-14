# 10 — Methodology

The protocol below is the actual contribution of this project. Every finding in
this directory is a consequence of applying it, and every retracted claim in
[finding 08](08-negative-results.md) is a consequence of not having applied it
yet.

It was not designed in advance. Each rule was added the day a result broke.

---

## The seven rules

### 1. Both halves, or it does not exist

A feature is real only if it works in the **first half and the second half** of
the sample. One good number over the whole period is what overfitting looks like
from the inside.

This is the cheapest filter available and it eliminated more candidates than
everything else combined. It is applied as a column in every screen:
`IC`, `p`, `hit%`, `H1 IC`, `H2 IC`, `verdict`.

### 2. One observation per event

If the event happens once a day, score it once a day. If a signal persists for
thirty minutes, collapse it into one trade before computing a hit-rate.

Violating this inflated results by **8–11×** and produced every false discovery
in [finding 02](02-measurement-illusions.md).

### 3. Costs are part of the hypothesis, not a post-hoc adjustment

A directional accuracy without a cost model is not a partial result — it is not
a result. The previous-day range breakout went from 60% accurate to a
−₹26,861 loss on the same data with no change other than applying costs.

Cost assumptions are stated explicitly and sensitivity is reported: the PCR
signal in [finding 07](07-edge-decay.md) is profitable below 14 points of cost
and dead above 16.

### 4. Beat the baseline, not zero

Every accuracy is reported against the relevant naive rule:

- direction → majority-class / always-long (~50%)
- volatility → persistence ("tomorrow looks like today", ~80% on BANKNIFTY)
- any feature → the price-only equivalent on the same days

BANKNIFTY volatility at 82% has an edge of +0.5 points. NIFTY 50 at 77% has an
edge of +9.3. Without the baseline column these look like the same result.

### 5. Regress out the obvious explanation

If a new feature correlates with something already known, test the **residual**.
`dist_maxpain` looked like an options edge at 60% until it turned out to be
0.818 correlated with the 10-day return — price mean reversion in an options
costume.

Correlated features are combined into a single z-score rather than counted as
separate discoveries.

### 6. Fix the rule before looking at the result

Thresholds set after seeing the outcome are not thresholds, they are fitted
parameters. In `research/test_pinning.py` the entry rule is fixed at 1σ in
advance and the training window is used only to set the z-score scale.

Where a threshold was deliberately left untuned to avoid snooping, this is
stated — as in the volatility-filtered ORB result in
[finding 04](04-volatility-predictability.md).

### 7. Report the recent window separately

A full-sample profit factor above 1.0 that was entirely earned in year one is a
dead signal reported as a live one. See [finding 07](07-edge-decay.md).

---

## The label problem

Worth separating out, because it caused a failure that none of the seven rules
would have caught.

A meta-filter trained on *"will the target be hit?"* improved its hit-rate from
42% to 53% — and made the money worse. It had learned to prefer small, easy
targets. Retrained on *"will this trade earn at least 5 points per minute?"*, the
hit-rate mattered less and the geometry stayed honest.

**The label encodes what you actually want.** A model optimises the label, not
the intention behind it. Accuracy on the wrong label is worse than no model,
because it is confidently wrong in a direction that looks like success.

---

## What honest reporting costs

Applying this protocol means:

- 100+ strategies tested, **zero** robust directional edges found;
- four previously documented positive results retracted;
- the project's headline number falling from 71% to ~51%;
- the one genuinely profitable signal found ([finding 07](07-edge-decay.md))
  discovered to be already dead.

That is the correct outcome, not a failure of the research. The alternative was
available at every step — keep the 71%, ship the ORB engine, sell the PCR
signal — and each one would have been a real loss to a real account.

---

## Reproducing

Every **REPRODUCIBLE** result regenerates from `research/`:

```bash
python3 -m research.backtest_direction     # forecast engine, per horizon and tier
python3 -m research.backtest_ict           # ICT liquidity-sweep, mechanised
python3 -m research.screen_positioning     # OI + participant feature screen
python3 -m research.test_pinning           # the pinning lead, out-of-sample
python3 -m research.test_oc_residual       # decade test, price regressed out
python3 -m research.test_intraday_oi       # intraday OI (accumulating)
```

Results marked **RECORDED** were produced by scratch scripts during live
research sessions and were not kept. Their numbers and dates are reported as
recorded and the method is described in enough detail to rebuild. They are
labelled so that a reader can weight them accordingly — a recorded result is
evidence, not proof.
