# 04 — Volatility is predictable, direction is not

**Claim tested.** That the *magnitude* of the next move is more forecastable
than its sign.

**Verdict.** Confirmed, and it is the strongest genuine result in the project —
but the honest version is narrower than the headline.

Through the same harness applied to direction: **70.8% (BANKNIFTY) / 72.0%
(NIFTY 50)** per bar, and **71.3% / 72.1%** per event. Shuffled-label controls
sit at 52–54%.

The decisive comparison is not against a coinflip, though. Volatility clusters,
so a persistence rule — "the next 30 minutes look like the last 30" — already
scores **67.4% / 67.4%**. The model's edge over that is **+3.3 / +3.7 points**
per event, and at n_eff the model and baseline intervals **overlap**. The edge
over persistence is real-looking but not established.

> Status: **REPRODUCIBLE** — `research/volatility_test.py`,
> [`results/volatility_test_30m.json`](../results/volatility_test_30m.json).

---

## Why it should work

Volatility clusters. Large moves follow large moves and quiet follows quiet —
one of the oldest and best-documented properties of financial time series, and
the reason the entire GARCH literature exists. Intraday there is a second,
equally robust structure: the U-shape, with volatility high at the open, low
through midday, and rising again into the close.

Neither of these says anything about direction. Both are strongly present in the
data we already have.

## The result that matters most

**Volatility survives per-event scoring; direction does not.**

| | per bar | per event |
|---|---|---|
| Direction (XGBoost, BANKNIFTY) | 51.3% | **50.4%** |
| Volatility (LightGBM, BANKNIFTY) | 70.8% | **71.3%** |

The same correction, on the same features and splits, removes one and leaves the
other untouched. That is the strongest available evidence that the correction in
[finding 12](12-corrected-estimator.md) separates signal from artefact rather
than flattening everything it touches.

The shuffled-label control tells the same story: for volatility the gap between
real and shuffled labels is ~18 points; for direction it was under 2.

## Earlier numbers, and how they compare

The production LSTM and an earlier gradient-boosted version are below. They are
consistent with the harness run in direction and magnitude, but were scored
per-bar without a persistence baseline, so their headline figures overstate what
is attributable to the model.

### LSTM (the shipped model)

24 five-minute bars of history → BIG vs QUIET over the next 30 minutes.

| Index | Out-of-sample accuracy | Stability |
|---|---|---|
| BANKNIFTY | **68%** | held in both halves |
| NIFTY 50 | **74%** | held in both halves |

> Status: **REPRODUCIBLE** — `oracle/models/volatility.py`; cached weights in
> `data_cache/{name}_vol_lstm.pt`.

### Gradient-boosted version (earlier, 90-day walk-forward)

| Index | Horizon | Raw accuracy | Edge over naive persistence |
|---|---|---|---|
| NIFTY 50 | 30m | 77% | **+9.3 pts** |
| NIFTY 50 | 60m | 81% | **+12.6 pts** |
| BANKNIFTY | 30–60m | 79–82% | +0.5 to +2.3 pts |

> Status: **REPRODUCIBLE** — `research/archive/volatility_engine.py`.

## The caveat that matters most

**High raw accuracy is not the same as edge.** Because volatility clusters, a
persistence rule already scores 67–68% at the 30-minute horizon in the harness
run, so the model's genuine contribution is the +3.3 to +3.7 points on top —
and at n_eff even that is not separated from the baseline.

The earlier "+9.3 / +12.6 points over baseline" figures used a different
persistence definition and a 90-day window; the harness numbers above supersede
them. Reporting a 72% without the baseline would be the same error this project
spent months finding in other people's numbers, and in its own.

This was a genuinely useful correction and it arrived from three independent
directions in the same week — an external review of the repository, a separate
meta-labelling analysis, and our own testing all converged on *predict magnitude,
not direction*. Convergence from independent sources is the closest thing to
confirmation available without new data.

## Volatility is a component, not a trade

A magnitude forecast does not by itself say what to buy. It has been tested in
two roles.

### As a filter on a directional strategy — partial, not enough

Opening-range breakout traded only on days a walk-forward classifier predicted
high post-09:45 volatility:

| | Unfiltered | Volatility-filtered |
|---|---|---|
| BANKNIFTY | −18.5R | **−9.2R** |
| NIFTY 50 | +0.9R | **+2.4R** (train +0.7R, test +1.7R) |

It halves the BANKNIFTY loss and produced the first configuration in the entire
investigation that was positive on *both* train and test. But BANKNIFTY still
loses, and the NIFTY 50 result rests on 58 trades after the filter cut 159 — the
train sample is 12 trades. That is inside the noise band. The threshold was
deliberately not tuned, to avoid manufacturing the result.

Recorded as a **research lead, not a signal**.

> Status: **RECORDED** (27 Jul 2026).

### As an options signal — the intended use

This is where a direction-free magnitude forecast is actually worth something,
because option structures are priced on magnitude:

| Forecast | Structure | Logic |
|---|---|---|
| BIG move | Long straddle / strangle | Profits on a large move either way |
| QUIET | Iron condor (defined risk) | Profits from decay if price stays in range |

`oracle/strategy/options.py` implements this with Black-Scholes breakevens,
max profit and loss, probability of profit and position sizing, taking India VIX
as the implied-volatility input.

**This has not been validated on real option premiums.** It is the correct
next experiment and it requires intraday option data the project does not yet
have. One important warning from [finding 05](05-variance-risk-premium.md)
applies in advance: when the same volatility model was used to filter premium
*selling*, it made the result worse, because premium is richest exactly when the
model predicts BIG. Predicting volatility correctly and being paid for it are
different problems.

## The honest framing

Volatility prediction at 71–72% is real and verified, and it survives the
correction that removed the directional result. It is also:

- **not directional** — it never says up or down;
- **mostly baseline** — persistence reaches 67–68%, and the model's margin over
  it is not established at the effective sample size;
- **not yet a validated trade** — the options layer built on it is untested on
  real premiums.

It is the most promising thread in the project and it is still a thread.
