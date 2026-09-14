# 04 — Volatility is predictable, direction is not

**Claim tested.** That the *magnitude* of the next move is more forecastable
than its sign.

**Verdict.** Confirmed, and by a wide margin. This is the strongest genuine
result in the project: **68–74% out-of-sample, stable across both test halves**,
against ~50% for direction on the identical data and features.

---

## Why it should work

Volatility clusters. Large moves follow large moves and quiet follows quiet —
one of the oldest and best-documented properties of financial time series, and
the reason the entire GARCH literature exists. Intraday there is a second,
equally robust structure: the U-shape, with volatility high at the open, low
through midday, and rising again into the close.

Neither of these says anything about direction. Both are strongly present in the
data we already have.

## Result

Binary target: will realised volatility over the next *k* bars exceed the
training-set median?

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
naive rule — "tomorrow looks like today" — already scores around 80% on
BANKNIFTY. Against that baseline the model adds almost nothing there. The real
result is NIFTY 50 at 30 and 60 minutes, where the model beats persistence by
9–13 points.

Reporting the 82% without the baseline would be the same error this project
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

Volatility prediction at 68–74% is real, verified, and stable. It is also:

- **not directional** — it never says up or down;
- **partly baseline** — persistence gets most of the way there on BANKNIFTY;
- **not yet a validated trade** — the options layer built on it is untested on
  real premiums.

It is the most promising thread in the project and it is still a thread.
