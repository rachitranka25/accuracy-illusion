# NIFTY ORACLE — Project Structure

Single-engine, honest intraday forecaster for BankNifty & Nifty50.
One forecast engine (`forecast_engine.py`), one dashboard, clear docs, and a
research sandbox. Full docs (PDF + HTML) live in `docs/`.

## Core (the live product)

| File | Role |
|---|---|
| `forecast_engine.py` | **THE engine.** Multi-horizon (5/15/30/60m) LightGBM forecaster, HIGH/MED/LOW confidence tiers, STRONG-signal gate, realistic (data-calibrated) targets. |
| `nifty_dashboard.py` | Web dashboard server (localhost:8181). Live/closed workers, logs forecasts to `forecast_history_v2.json`. `H2H=1` enables the opt-in legacy head-to-head logger. |
| `forecast_feed.html` | Main forecast-feed page (`/forecasts`). |
| `dashboard.html` | Landing dashboard (`/`). |
| `quantum_live.html` | Alt market-state view (`/quantum`). |
| `angel_data.py` | Angel One SmartAPI feed (TOTP auto-login, works abroad). Yahoo fallback. |
| `nifty_live.py` | Shared indicators + data helpers + model factory. |
| `ai_analyst.py` | AI commentary (NVIDIA Nemotron primary, Gemini fallback). |
| `envconfig.py` | Loads secrets from `.env` first, JSON config fallback. |

## Tooling

| File | Role |
|---|---|
| `train_models.py` | (Re)train the 8 forecast models. |
| `backtest.py` | Honest walk-forward backtest (no lookahead, cost-aware). |
| `fetch_futures.py` | Pull NIFTY/BANKNIFTY **futures** OHLC + **real volume** from Angel (daily ~2.5yr, 5-min ~3mo). |
| `fetch_bhavcopy.py` | Bulk-download NSE F&O bhavcopy → 2yr **options + futures EOD with Open Interest** (free). |
| `requirements.txt` | Python deps. |

## data_cache/
- `*_5m_long.csv` — 17 mo of 5-min index bars (volume = 0; index spot has no volume).
- `*_FUT_daily.csv` / `*_FUT_5m.csv` — futures with **real volume** (fixes the fake-VWAP problem).
- `*_options_eod.csv` — **2 yr of per-strike option EOD**: OHLC, volume, **OI, ΔOI** (846k Nifty / 536k BankNifty rows). No IV (compute from premium).
- `*_futures_eod.csv` — index futures EOD from bhavcopy.
- `*_models_v2.pkl` — trained forecast models.

## docs/
Technical + data/training documentation (PDF + HTML), kept current with findings.

## experiments/  (research — NOT in the live path)
| File | Status |
|---|---|
| `volatility_engine.py` | Realized-vol regime predictor. Held on 90-day; the most predictable target (~54-65% per horizon) but it's magnitude, not direction — a component, not a trade. |
| `vrp_straddle_test.py` | Volatility-risk-premium test: short ATM straddle. Real edge (62-72% win) but retail cost + tail risk (Taleb distribution) kill it. |
| `oi_signals_test.py` | ~17 OI/options next-day signals. Best = Nifty PCR-extremes (58%) — but it **decayed and is dead** in the last 6 months. |
| `best_ensemble_test.py` | Best selective multi-signal ensemble. Robust ceiling ~52-54% out-of-sample. |
| `volatility_forecast_test.py` | Per-horizon volatility prediction (~54-65%). |
| `legacy_engine.py` | Old "Beast" (VIX + cross-index). No persistent edge vs the main engine. |
| `meta_labeling.py` | Triple-barrier + meta-labeling. Tested twice, collapsed out-of-sample. |
| `straddle_backtest.py` | Cost-aware short-straddle template. |
| `nifty_oracle.py` | Original standalone ORB/daily-bias script (superseded). |
| `head2head.json` | Live new-vs-legacy log (written when the dashboard runs with `H2H=1`). |

## Honest status (backed by theory + our own empirical tests)
Directional accuracy sits ~50-58% (regime/selectivity dependent); volatility
prediction ~54-65%. Across 100+ tested strategies (price, OI/PCR, VRP, regime,
ensemble, volatility) **no configuration produces a robust tradeable edge after
costs** — the one positive edge found (PCR-extremes) decayed to death, and the
real structural edge (VRP) is eaten by retail costs + tail risk. This matches
the academic literature exactly. Treat the system as an honest information /
learning / paper-trading tool, **not** an income source.

## Security
Secrets live **only** in `.env` (git-ignored). Never zip/share `angel_config.json`
— the Angel TOTP secret is a live 2FA bypass. If it was ever shared, rotate the
API key and reset TOTP.
