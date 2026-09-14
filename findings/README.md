# Findings

This directory is the research record: every question the project asked, the
test that answered it, and the number that came back — including, especially,
the ones that came back negative.

It is written to be read in order. Each document states a **claim**, the
**method** used to test it, the **result**, and the **caveat** that keeps the
result honest.

| # | Finding | Verdict |
|---|---|---|
| [01](01-directional-ceiling.md) | Intraday direction on NSE index 5-minute bars | **No extractable signal.** ~50% out-of-sample across every model family tested. |
| [02](02-measurement-illusions.md) | Why our own screens repeatedly showed 60–85% | **Measurement artefacts.** Overlapping samples inflate per-minute hit-rates 8–11×. |
| [03](03-cost-geometry.md) | Does a ~53% direction call survive costs? | **No.** The typical move is the same size as the round trip. |
| [04](04-volatility-predictability.md) | Is *magnitude* more predictable than *direction*? | **Yes — the strongest real result.** 68–74% out-of-sample, stable. |
| [05](05-variance-risk-premium.md) | Is there a retail-reachable profitable edge at all? | **One.** Short NIFTY ATM premium: +22.5 pts/trade after cost. With a fat left tail. |
| [06](06-open-interest.md) | Does the option chain know something price does not? | **A little.** IC ≈ 0.04–0.06 over 2,181 days, stable — real but small. |
| [07](07-edge-decay.md) | Do the edges that *do* appear survive? | **No.** The best one decayed from 74% to 38% in under two years. |
| [08](08-negative-results.md) | Complete register of what was tested and failed | 100+ strategies, 7 model families, 3 data regimes. |
| [09](09-data-assets.md) | What can a retail account actually obtain for free? | More than expected: a decade of option chains, 2 years of OI, real volume. |
| [10](10-methodology.md) | The protocol that killed the mirages | Out-of-sample + both-halves stability + per-event scoring + costs. |

## How to read the numbers

Three labels appear throughout:

- **REPRODUCIBLE** — a script in [`../research/`](../research/) regenerates the
  number from data in `data_cache/`. Cited with the script name.
- **RECORDED** — the result was produced during a live research session by a
  scratch script that was not kept. The number and date are reported as
  recorded; the method is described so it can be rebuilt.
- **LIVE** — observed on a running session, not a backtest. Single-session live
  numbers are the *least* reliable evidence in this document and are shown only
  where they illustrate variance.

## The one-line version

Direction over short horizons is a coinflip that costs money to play. Magnitude
is genuinely forecastable. The only edge we could both measure and reach is
selling variance, and it pays in small regular amounts while risking large
irregular ones.
