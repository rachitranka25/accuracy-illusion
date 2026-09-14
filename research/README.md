# Research

One script per finding. None of these are imported by the live system.

Run from the repository root as modules so package imports resolve:

```bash
python3 -m research.test_oc_residual       # decade test: does the OI residual predict anything?
python3 -m research.test_pinning           # fade-the-stretch rule, fixed in advance, out-of-sample
python3 -m research.screen_positioning     # one-at-a-time screen of OI + participant features
python3 -m research.test_intraday_oi       # intraday OI (accumulating; no result yet)
python3 -m research.backtest_direction     # forecast engine accuracy per horizon and tier
python3 -m research.backtest_ict           # ICT liquidity-sweep reversal, mechanised with costs
python3 -m research.train_models           # (re)train the multi-horizon direction models
```

Every screen reports `IC`, `p`, `hit%`, `H1 IC`, `H2 IC` and a verdict. A feature
is only kept if it holds in **both halves** of the sample — see
[`../findings/10-methodology.md`](../findings/10-methodology.md).

## archive/

Earlier experiments, kept because a research log that deletes its failures is
worthless. Each is referenced from the findings that cite it.

| Script | What it showed |
|---|---|
| `volatility_engine.py` | Gradient-boosted volatility regime model. Held on a 90-day walk-forward — NIFTY 50 30m 77% (+9.3 over baseline), 60m 81% (+12.6). |
| `vrp_straddle_test.py` | Short ATM straddle on real EOD premiums. NIFTY +22.5 pts/trade after cost; BANKNIFTY −7.3. |
| `oi_signals_test.py` | ~17 OI/options next-day signals. Best was NIFTY PCR-extremes — profitable in 2024, dead by 2026. |
| `meta_labeling.py` | Triple-barrier + meta-labelling (purge + embargo). 58.5% at 45 days, 25% at 90. Collapsed. |
| `legacy_engine.py` | The earlier VIX + cross-index "Beast". No persistent edge; its apparent gain was a train/serve skew. |
| `best_ensemble_test.py` | Best selective multi-signal ensemble. Ceiling ~52–54%. |
| `volatility_forecast_test.py` | Per-horizon volatility prediction, ~54–65%. |
| `straddle_backtest.py` | Cost-aware short-straddle template. |
| `nifty_oracle.py` | The original standalone ORB / daily-bias script. Superseded; ORB has no robust edge. |
| `pulse90_experiment.py` | Attempt to push the Pulse window's green rate. Retained as a record of what selectivity alone can and cannot do. |
