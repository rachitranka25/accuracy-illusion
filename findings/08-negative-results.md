# 08 — Register of negative results

Everything tested and rejected, with the reason. This is the largest section of
the research record and the most useful part of it: it is the list of things
nobody needs to try again on this data.

A negative result is only worth anything if the test was capable of returning a
positive. Each entry states the protocol.

---

## Model architectures

| Model | Task | Result |
|---|---|---|
| Gradient Boosting | 30m direction | ~50% OOS |
| LightGBM | 30m direction | 49–53% OOS |
| XGBoost | 30m direction | ~50% OOS |
| LSTM | 30m direction | 49% OOS |
| Transformer | 30m direction | 51% OOS |
| Temporal Fusion Transformer | 30m direction | 46% OOS |
| k-Nearest Neighbours | 30m direction | 50% OOS |

Ceiling test (over-capacity model, shuffled-label control): 96% train / 49% test,
shuffled labels also 50%. See [finding 01](01-directional-ceiling.md).

## Price-based strategies

| Strategy | Result |
|---|---|
| Opening-range breakout (ORB) | BANKNIFTY **−18.6R** over 351 sessions; tuned + adaptive still −12.6R. Loses on train, positive only on recent test = regime luck. |
| Rapid-fire 5-minute ML scalper | Lost in **all 18** configurations tested. |
| Momentum, VWAP-trend, supply/demand zones | ~50–54%, none stable. |
| Gap-fade | 62% per bar → **44–48% per day**. Illusion. |
| Previous-day range breakout | 60% directional → 51% profitable, PF 0.95, −₹26,861. |
| Global overnight lead | Predicts the **gap** at 62–65%, but intraday reverts to ~48%. The gap has already happened at the open — not tradeable. |
| ICT liquidity-sweep reversal | Mechanised with fixed rules and costs; no edge over coinflip. `research/backtest_ict.py` |
| 52-strategy vectorised sweep | Nothing above ~53–54% with both-halves stability. |

**A pattern worth naming:** the strategies that failed most clearly — ORB,
momentum, VWAP, gap-fade — are the most popular retail strategies. They fail
*because* they are crowded. Edge does not survive in the places everyone looks.

## Features

| Feature set | Result |
|---|---|
| Candlestick pattern flags as model inputs | No improvement, two independent tests. |
| Chart-pattern "essence" (swing highs/lows, range position, HH/LL counts, Bollinger squeeze, choppiness) | BANKNIFTY flat; NIFTY 50 mixed (5m +3.7, 30m −5.5) = overfitting noise. |
| Cross-index, India VIX, cointegration extras ("legacy" engine) | Appeared to lift BANKNIFTY +2 to +4.5 points. **Retracted** — the gain came from a train/serve skew: features were built from two years of continuous history at training time but only today's intraday bars at serve time. Head-to-head over a live session confirmed no persistent edge. |
| Volume-flow / CVD from real futures volume | BANKNIFTY 52% → 55% hint, small sample, not confirmed. |

Patterns remain in the interface as visual confirmation. They are not model
inputs.

## Meta-labelling

Tested three times. Two failures and one qualified success that still lost money.

| Attempt | Result |
|---|---|
| Triple-barrier + meta-labelling, multi-horizon | Base game deeply negative (~−0.9R); meta could not find enough good trades. |
| Proper López de Prado implementation (AFML Ch. 3/7, purge + embargo) | NIFTY 50 60m at meta ≥ 0.60: **58.5% at 45 days → 25% at 90 days.** Collapsed, exactly as its author warned. |
| Points-per-minute label, BANKNIFTY | **Held out-of-sample** over 106 days and improved quality 26% — and still totalled −981. See [finding 03](03-cost-geometry.md). |

A metric warning from the second attempt: scoring "accuracy versus triple-barrier
truth" counts flat bars as wrong, so baselines look like 37% and improvements
look like +18 points. The absolute out-of-sample collapse is the truth; the delta
is an artefact of the denominator.

A finding that emerged from this work and is worth keeping: at the 15-minute
horizon **~72% of bars have no ±1.5σ move at all.** Short-horizon "direction
accuracy" is therefore mostly capturing drift, not signal — which independently
confirms [finding 03](03-cost-geometry.md).

## Data quality

| Issue | Status |
|---|---|
| Index spot has **volume = 0** | Confirmed. Every "VWAP" computed on index spot is an unweighted expanding mean of typical price, not a volume-weighted average. Real volume exists only on futures (`oracle/data/futures.py`). Whether true VWAP helps is **untested**. |
| Majority-class baseline check | Base UP rate ≈ 50% at the 5-minute level. The model's small HIGH-tier edge (+1 to +3.4 points) is real-but-small, not a masked bull bias. |
| Angel revises the final bar | Median difference versus official candles: 0.0 points. The last bar is revised to the official close. Not an error. |
| Broken by automated agents | `forecast_engine` was left unrunnable (undefined names at construction) and a backtest script called a removed method. Both rebuilt. Cited as a reason every claim in this repository carries a reproduction path. |

## Infrastructure rejected

| Item | Reason |
|---|---|
| Upstox integration | Requires daily OAuth with SMS OTP to an Indian number. Unusable from abroad. Deleted. Angel One works because it uses TOTP, which is local. |
| NSE live option-chain API | Geo-blocked (403/404). The static bhavcopy archive is **not** — see [finding 09](09-data-assets.md). |
| Historical order-flow / Level-2 | Paid only. NSE and SEBI do not distribute it free. Live collection started instead. |
| Intraday per-strike OI, retroactively | Paid only (Stolo, TrueData). Forward collection started instead. |

## Things that were true and stopped being true

Recorded because a research log that quietly edits its own history is worthless:

1. **"ORB is the positive-expectancy money engine."** Based on a 60-day Yahoo
   window. The full 351-session backtest reversed it. Documentation corrected.
2. **"VIX and cross-index extras lift BANKNIFTY by 2–4.5 points."** Train/serve
   skew. Retracted; features removed.
3. **"71% accuracy."** Per-minute counting on one trending Friday. Traced to
   ~51% when scored correctly. See [finding 02](02-measurement-illusions.md).
4. **"PCR extremes are a profitable edge."** True for 2024. Dead by 2026. See
   [finding 07](07-edge-decay.md).
5. **"Meta-labelling reaches 62–65%."** Quoted from an external analysis; not
   supported by either of our implementations.
