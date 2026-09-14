# The Accuracy Illusion — paper outline

Format-agnostic. Every claim is listed with the artefact that supports it, so
the draft can be poured into any venue's template without re-deriving anything.

**Status: evidence complete for the core sections. Gaps are listed at the
bottom and are honest gaps, not formatting work.**

---

## Title

**The Accuracy Illusion: How Overlapping Samples Manufacture 71% Hit-Rates in
Intraday Index Forecasting**

## Abstract — the four sentences it has to contain

1. Every-minute multi-horizon directional forecasting systems report session
   hit-rates that are not valid estimates of anything, because overlapping
   outcome windows combine with persistent predictions to inflate the variance
   of the reported rate far beyond what its stated sample size implies.
2. On NSE index futures we measure this directly: a forecaster with *exactly
   zero skill* posts a ≥60% session once in six and a ≥71% session once in 51,
   with a stated confidence interval 1.66× too narrow — and across a panel of
   two indices and four horizons, a ≥71% reading appears every seventh session.
3. Under correct scoring, nine model families on identical purged walk-forward
   splits reach at most 51.3% out-of-sample, and 86 rule-based strategies
   produce a best Sharpe *below* what a block-bootstrap null generates from
   selection alone.
4. We then show the result that matters economically: break-even accuracy
   derived from measured move distributions is 71.5% at the 30-minute horizon
   and exceeds 100% at shorter horizons and at option-level costs — so the
   small, statistically real edge that does exist is an order of magnitude too
   small to trade.

## 1. Introduction

**Claims and evidence**

| Claim | Support |
|---|---|
| High reported accuracies are common in this field and drive real capital allocation | Literature + the project's own history (5 retracted claims, `FINDINGS.md`) |
| Our own system produced 71% and believed it for months | `findings/02`, traced to source |
| The error is in the denominator, not the model | `research/overlap_simulation.py` |

**Contributions, stated narrowly enough to defend**

1. A measurement of the overlapping-sample artefact *as it occurs in practice* —
   combined with prediction persistence taken from a live system's own log, and
   with panel multiplicity.
2. A complete control set for a negative result: shuffled labels, injected
   positive controls, nine architectures, block-bootstrap null.
3. The required-versus-achieved accuracy framing, with break-even computed from
   measured move distributions.
4. A reusable free-data corpus for NSE index derivatives.

## 2. Related work

Per `paper/REFERENCES.md`. Three threads:

- **Overlapping observations in financial econometrics** — known; we measure a
  specific practical consequence. *This section must not overclaim.*
- **Backtest overfitting and multiple testing** — DSR, PBO/CSCV. We apply both,
  and report where each fails in our setting.
- **Intraday predictability and microstructure** — the ceiling literature, and
  Kolm et al. for where real short-horizon alpha lives.

## 3. Data

`findings/09-data-assets.md`, `oracle/data/`.

| Asset | Coverage |
|---|---|
| Index 5-min bars, BANKNIFTY / NIFTY 50 | 17 months, 26,265 bars each |
| Option chain EOD with OI | 2 years, 1.38M rows |
| Long-history option chain | **10 years**, 2,181 days |
| Participant-wise OI | 2 years, 515 days |
| Futures with real volume | 2.5 years daily |

Three points worth a paragraph each: index spot prints volume = 0 (so
spot "VWAP" is not a VWAP); the NSE bhavcopy archive is not geo-blocked while
the live API is; TOTP authentication makes the data pipeline location-independent.

## 4. Method

`findings/10-methodology.md`, `research/common.py`.

Purged expanding walk-forward with embargo · exact Clopper-Pearson intervals ·
Pesaran-Timmermann · shuffled-label controls on every model · injected positive
controls · one observation per event · costs inside the hypothesis.

## 5. The mechanism — **the core section**

`research/overlap_simulation.py`, `research/ceiling_test.py`

- 5.1 Overlap is arithmetic: ACF 0.84 → 0.52 → 0.04, dying at exactly lag *k*;
  confirmed independently at *k* = 6 and *k* = 12. **Figure 1.**
- 5.2 Persistence is empirical: mean run 15.2 min, longest 135, measured on the
  live log. Neither ingredient inflates anything alone — state this, since our
  first simulation got it wrong and found nothing.
- 5.3 The measured distribution of a zero-skill forecaster. **Figure 2.**
  Note the multimodality: ~12 independent decisions per session, not 69.
- 5.4 Calibration: 1.66× understatement per-bar versus 0.99× scored per event.
- 5.5 Multiplicity: 2 indices × 4 horizons → a ≥71% session every 7 days.
- 5.6 Case study: our own 71%, traced to one Friday.

## 6. What survives correct scoring

`research/model_zoo.py`, `research/ceiling_test.py`, `research/strategy_sweep.py`

- 6.1 The ceiling test: 100% train / 50.5% test / 50.2% shuffled; positive
  controls recovered. The pipeline can find signal; there is none.
- 6.2 Nine families. **Figure 4.** Best 51.3% (BANKNIFTY, XGBoost, p = 0.001,
  survives Bonferroni across 18 tests) — **and it does not replicate on
  NIFTY 50.** Report the non-replication prominently; it is the honest reading.
- 6.3 86 strategies: 2 positive Sharpes, 0 significant. Empirical null's
  best-of-86 median 0.44 against 0.21 observed, P = 0.658.

## 7. Economic significance — **the section that decides the paper**

`research/cost_geometry.py`. **Figure 3.**

- 7.1 `p* = ½ + c/2m`, and why c/m is the whole game.
- 7.2 Measured: 71.5% needed at 30m, >100% at 5m on both indices.
- 7.3 Cost regimes: at ATM option cost the requirement is unreachable at any
  accuracy.
- 7.4 The reframing: a genuine 85% clears futures break-even by 13 points and
  still loses on options. So an 85% claim is either an artefact or worth far
  less than it sounds.
- 7.5 Corroboration: the meta-filter that verifiably improved trade quality 26%
  out-of-sample and still lost money.

## 8. What is predictable instead

`findings/04`, `findings/05`, `findings/06`

- 8.1 Volatility: 68% / 74% OOS, stable — **with the baseline caveat** that
  persistence already scores ~80% on BANKNIFTY, so the real edge is NIFTY 50.
- 8.2 Variance risk premium: +22.5 pts/trade after cost, 61% win, and a fat left
  tail (worst −847 NIFTY, −2,002 BANKNIFTY). Structural, not predictive.
- 8.3 Option-chain residuals over 2,181 days: IC 0.04–0.06, stable, where
  price-only features show nothing. Flow beats level.

## 9. Limitations — write this section first, not last

- Two indices, one market, 17 months of intraday bars.
- The symmetric-bracket break-even model is a modelling choice; asymmetric
  payoffs change the arithmetic in both directions.
- The TFT is a compact variant, not the published architecture.
- Several results in `findings/` are **RECORDED**, not reproducible, and are
  labelled as such throughout. They are not load-bearing for any claim in
  sections 5–7.
- Volume is absent from index spot, so any VWAP-derived feature is misnamed.
- The order-flow hypothesis — the one place the literature locates real
  short-horizon alpha — is **untested here** for want of history.

## 10. Conclusion

The field's headline metric is not merely noisy — it is systematically biased
upward by the way it is computed, and the bias is large enough to manufacture
the exact numbers used in marketing. Correct scoring leaves a real but tiny
directional effect, an order of magnitude below its own break-even threshold.
What is left that works is not direction at all.

---

## Evidence inventory

| Section | Script | Results file | Figure |
|---|---|---|---|
| 5.1 | `research.ceiling_test` | `ceiling_test_30m.json`, `_60m.json` | Fig 1 |
| 5.2–5.5 | `research.overlap_simulation` | `overlap_simulation_30m.json` | Fig 2 |
| 6.1 | `research.ceiling_test` | `ceiling_test_*.json` | — |
| 6.2 | `research.model_zoo` | `model_zoo_30m.json` | Fig 4 |
| 6.3 | `research.strategy_sweep` | `strategy_sweep_30m.json` | — |
| 7 | `research.cost_geometry` | `cost_geometry.json` | Fig 3 |
| 8.3 | `research.test_oc_residual` | (console) | — |

## Open gaps

Honest list. These are missing work, not missing prose.

1. **Model zoo at 5m / 15m / 60m.** Only 30m is complete (60m running). The
   cost-geometry table has an "achieved" column that is empty at three of four
   horizons, and 60m is where the break-even requirement is lowest — so it is
   the horizon most worth filling.
2. **A third and fourth index.** FINNIFTY, MIDCPNIFTY and SENSEX bars are on
   disk and unused. Replication across five indices would materially strengthen
   section 6.2, especially given that BANKNIFTY's edge does not replicate on
   NIFTY 50.
3. **The overlapping-returns econometrics literature** must be engaged with
   directly (`REFERENCES.md`), or the contribution in section 5 is overstated.
4. **The pinning lead** (`research/test_pinning.py`, 60% over 55 OOS days) should
   be re-run on the decade of chain data. Either it survives at n ≈ 500 and
   becomes a genuine positive result, or it dies and joins section 6 — both
   outcomes improve the paper.
5. **Citations flagged unverified** in `REFERENCES.md` must be resolved or the
   corresponding sentences removed.
6. **`results/` for the sensitivity of the break-even model** to asymmetric
   payoffs — currently the symmetric case only.
