# The Accuracy Illusion

**How overlapping samples manufacture 71% hit-rates in intraday index forecasting.**

A fourteen-month empirical study of what can and cannot be predicted on NSE
index derivatives using only data a retail account can obtain for free —
together with the forecasting system built to test it.

The short version: **every high accuracy number this project produced turned out
to be a counting error**, and the small real edge that survives correct counting
is an order of magnitude too small to pay for the round trip. Magnitude, unlike
direction, is genuinely forecastable. The full record is in
[`FINDINGS.md`](FINDINGS.md).

---

## The result

| | |
|---|---|
| Best directional accuracy, 9 model families, purged walk-forward | **51.3%** (p = 0.001) |
| Same pipeline on **shuffled random labels** | **49.7%** |
| Accuracy **required** to break even at 4 bp cost | **71.5%** |
| Accuracy required at ATM **option** costs | **impossible** (cost > median move) |
| How often a **zero-skill** forecaster posts a ≥71% session | **1 in 7** |
| Confidence-interval understatement from per-bar scoring | **1.66×** |
| Rule-based strategies profitable after multiple-testing correction | **0 of 86** |
| Volatility (magnitude) prediction, out-of-sample, stable | **68–74%** |
| Profitable structures found | **1** — short NIFTY ATM premium, +22.5 pts/trade |

The central mechanism needs two ingredients, and neither is enough alone. A
30-minute forecast emitted every minute shares 29 of its 30 minutes of outcome
with the next one — **and** a real model repeats the same call for long stretches
(measured on this system's own live log: mean run 15.2 minutes, longest 135).
When both the call and the outcome persist, one trending afternoon is recorded as
dozens of hits.

Simulated against real bars, a forecaster with exactly zero skill reaches ≥60% on
one session in six and ≥71% on one in 51 — and a dashboard watching 2 indices ×
4 horizons sees a ≥71% reading **every seventh session**. This project's own
headline 71% was exactly that: one index, one horizon, one trending Friday.

The second half of the result is that even a genuine edge would not be enough.
The models do find something real — a 1.6-point lift over the shuffled control,
significant at n = 16,856 and surviving Bonferroni. It loses money on every
trade, because the break-even requirement is 20 points higher.

## Findings

Eleven documents, in reading order:

| # | | |
|---|---|---|
| 01 | [The directional ceiling](findings/01-directional-ceiling.md) | Nine model families, shuffled-label controls, and a positive control that proves the pipeline works. |
| 02 | [Measurement illusions](findings/02-measurement-illusions.md) | How 71%, 85% and 62% were each manufactured — simulated and measured. |
| 03 | [Cost geometry](findings/03-cost-geometry.md) | The accuracy that would actually be required: 71.5%, or impossible. |
| 04 | [Volatility is predictable](findings/04-volatility-predictability.md) | The strongest genuine result. |
| 05 | [The variance risk premium](findings/05-variance-risk-premium.md) | The one reachable edge, and its tail. |
| 06 | [Open interest](findings/06-open-interest.md) | A decade test: small, real, stable. |
| 07 | [Edges decay](findings/07-edge-decay.md) | The best signal found was already dead. |
| 08 | [Negative results](findings/08-negative-results.md) | The full register of what failed. |
| 09 | [Data assets](findings/09-data-assets.md) | What is actually free, and what is not. |
| 10 | [Methodology](findings/10-methodology.md) | The seven rules that killed the mirages. |
| 11 | [Multiple testing](findings/11-multiple-testing.md) | 86 strategies, and why the winner is worse than the null. |

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
  common.py             shared harness: purged splits, exact CIs, PT and DM tests
  archive/              earlier experiments, kept for the record
results/                committed raw output from every reproduction script
findings/               the research record
docs/                   technical documentation
```

## Reproducing

The five experiments behind the headline numbers, from the repository root:

```bash
python3 -m research.ceiling_test        # is there anything to predict at all?
python3 -m research.model_zoo           # 9 model families, identical purged splits
python3 -m research.overlap_simulation  # how a coinflip posts a 71% session
python3 -m research.strategy_sweep      # 86 strategies + multiple-testing correction
python3 -m research.cost_geometry       # the accuracy that would actually be needed
```

And the option-chain and positioning work:

```bash
python3 -m research.test_oc_residual       # decade test, price regressed out
python3 -m research.test_pinning           # the open lead, out-of-sample
python3 -m research.screen_positioning     # OI + participant feature screen
python3 -m research.backtest_direction     # forecast engine, per horizon and tier
python3 -m research.backtest_ict           # ICT liquidity sweep, mechanised
```

Fixed seed (`research/common.py`), native thread pools pinned to one thread.
Raw output is committed in [`results/`](results/).

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
claims, five of which are formally retracted in
[`FINDINGS.md`](FINDINGS.md#what-was-retracted).

## License

MIT — see [LICENSE](LICENSE).
