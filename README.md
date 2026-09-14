# The Accuracy Illusion

**How overlapping samples manufacture 71% hit-rates in intraday index forecasting.**

A fourteen-month empirical study of what can and cannot be predicted on NSE
index derivatives using only data a retail account can obtain for free —
together with the forecasting system built to test it.

The short version: **every high accuracy number this project produced turned out
to be a counting error.** Recounted correctly, intraday direction is a coinflip.
Magnitude is not. The full record is in [`FINDINGS.md`](FINDINGS.md).

---

## The result

| | |
|---|---|
| Directional accuracy, out-of-sample, 7 model families | **~50%** |
| Same pipeline on **shuffled random labels** | **50%** |
| Inflation factor from scoring overlapping forecasts | **8–11×** |
| Volatility (magnitude) prediction, out-of-sample, stable | **68–74%** |
| Strategies tested with a robust directional edge after costs | **0 of 100+** |
| Profitable structures found | **1** — short NIFTY ATM premium, +22.5 pts/trade |

The central mechanism: a 30-minute forecast emitted every minute shares 29 of
its 30 minutes of outcome with the next one. They are not independent
observations — they are one observation sampled thirty times. A single trending
afternoon is recorded as thirty hits. Collapse them into distinct trades and a
48–68% per-minute hit-rate becomes 26%.

This project's own headline figure of 71% was one trending Friday counted
per-minute. Scored correctly it is ~51%.

## Findings

Ten documents, in reading order:

| # | | |
|---|---|---|
| 01 | [The directional ceiling](findings/01-directional-ceiling.md) | The information is not in the data — proven independently of the model. |
| 02 | [Measurement illusions](findings/02-measurement-illusions.md) | How 71%, 85% and 62% were each manufactured. |
| 03 | [Cost geometry](findings/03-cost-geometry.md) | Why a 53% call still loses money. |
| 04 | [Volatility is predictable](findings/04-volatility-predictability.md) | The strongest genuine result. |
| 05 | [The variance risk premium](findings/05-variance-risk-premium.md) | The one reachable edge, and its tail. |
| 06 | [Open interest](findings/06-open-interest.md) | A decade test: small, real, stable. |
| 07 | [Edges decay](findings/07-edge-decay.md) | The best signal found was already dead. |
| 08 | [Negative results](findings/08-negative-results.md) | The full register of what failed. |
| 09 | [Data assets](findings/09-data-assets.md) | What is actually free, and what is not. |
| 10 | [Methodology](findings/10-methodology.md) | The seven rules that killed the mirages. |

## The system

A live multi-horizon forecasting dashboard for BANKNIFTY, NIFTY 50 and FINNIFTY.
It exists to generate the evidence above under realistic conditions, not to
trade.

```bash
pip install -r requirements.txt
cp .env.example .env          # add your own broker credentials
python3 run.py                # → http://localhost:8181
```

| Page | Shows |
|---|---|
| `/` | Live chart, paper trading, P&L |
| `/forecasts` | Per-minute multi-horizon forecasts with calibrated confidence tiers |
| `/pulse` | High-selectivity single-call view with honest green-rate scoring |
| `/volatility` | BIG / QUIET regime forecast and the option structures it implies |
| `/quantum` | Market-state indicator panel |

Market data comes from Angel One SmartAPI, which authenticates by locally
computed TOTP and therefore works from anywhere; Yahoo Finance is the fallback.
Credentials live only in `.env`.

## Layout

```
oracle/                 the system
  data/                 broker + exchange collectors (7 free sources)
  features/             indicators, positioning features
  models/               direction, volatility (LSTM), meta-filter
  strategy/             option structures, participant positioning
  app/                  dashboard server + web pages
research/               one script per finding — the reproduction path
  archive/              earlier experiments, kept for the record
findings/               the research record
docs/                   technical documentation
```

## Reproducing

```bash
python3 -m research.test_oc_residual       # decade test, price regressed out
python3 -m research.test_pinning           # the open lead, out-of-sample
python3 -m research.screen_positioning     # OI + participant feature screen
python3 -m research.backtest_direction     # forecast engine, per horizon and tier
python3 -m research.backtest_ict           # ICT liquidity sweep, mechanised
```

Results in `findings/` are labelled **REPRODUCIBLE** (a script here regenerates
them), **RECORDED** (produced by a scratch script during a live session; number
and method reported, script not kept) or **LIVE** (observed on a running
session — the weakest evidence, shown only to illustrate variance).

Market data is not committed. See
[`findings/09-data-assets.md`](findings/09-data-assets.md) for how to rebuild
every dataset from free sources.

## Disclaimer

This is a research repository. It is not investment advice, it is not a
registered advisory service, and none of it should be traded with money that
matters. Its central finding is that the strategies people sell as profitable
are mostly measurement errors — which applies to this repository's own earlier
claims, four of which are formally retracted in
[`FINDINGS.md`](FINDINGS.md#what-was-retracted).

## License

MIT — see [LICENSE](LICENSE).
