# NIFTY ORACLE — free intraday forecast engine (BankNifty + Nifty50)

Free intraday forecasting for BankNifty & Nifty50. Real-time data via **Angel
One SmartAPI** (auto-login, works from anywhere — no SMS OTP), with Yahoo
Finance as a fallback. No paid data, no subscriptions.

> **Read the honest truth section at the bottom before trusting any signal.**
> Realistic accuracy is ~48–59%, never 90%+. Anyone promising more is lying.
> A same-day cost-aware test (28 Jul 2026) lost money on both indices.

**One forecast engine** (`forecast_engine.py`). Project map: see `STRUCTURE.md`.
Full docs (PDF + HTML) in `docs/`. Archived research in `experiments/`.

## Quick start

```bash
cd ~/Desktop/QUANTSSSS
python3 nifty_dashboard.py          # the web dashboard → http://localhost:8181
```

Angel One logs in automatically (TOTP). Market open → **LIVE** mode; closed →
shows the last real session statically.

| Page | What it shows |
|---|---|
| `http://localhost:8181/` | **The Beast** — live chart, auto paper-trading, P&L, trade log |
| `http://localhost:8181/forecasts` | **Forecast Window** — every-minute multi-horizon forecast + the ⭐ STRONG / WAIT signal |
| `http://localhost:8181/quantum` | Market-state view — 7 indicator chips (RSI, EMA, MACD, VWAP, Bollinger, Stochastic, Volatility) |

## Commands

| Command | What it does |
|---|---|
| `python3 nifty_dashboard.py` | The web dashboard (above). |
| `H2H=1 python3 nifty_dashboard.py` | Same, **plus** the opt-in head-to-head logger (new vs the archived legacy engine → `experiments/head2head.json`). Off by default. |
| `python3 backtest.py` | Honest walk-forward accuracy of the forecast engine (per horizon + per tier). |
| `python3 train_models.py [--fresh]` | Pre-train / retrain the 8 forecast models. |
| `python3 fetch_futures.py` | Pull futures OHLC + **real volume** from Angel (daily ~2.5yr, 5-min ~3mo). |
| `python3 fetch_bhavcopy.py` | Bulk-download 2yr of NSE F&O bhavcopy → per-strike option **OI/volume/OHLC** EOD. |

Most accept `--capital 100000 --risk 1.0`.

## How the forecast works (single engine)

**`forecast_engine.py`** — every minute, for each index, it predicts direction
over 4 horizons (5m / 15m / 30m / 60m) using 8 LightGBM models trained on ~17
months of 5-minute data. Each forecast gets a confidence **tier**: HIGH /
MEDIUM / LOW (calibrated on held-out data).

- **⭐ STRONG signal** fires only when **3–4 of the 4 horizons are HIGH tier and
  agree** on direction — *and* it is **not** the midday dead-zone (11:00–13:30,
  where the model historically drops to ~48%). STRONG ≈ 54–59% historically.
- Every other minute shows **WAIT** — sitting out is a valid decision.

`nifty_live.py` is a shared utility module (indicators, data helpers, model
factory) used by the dashboard; its terminal ORB modes are kept only for
reference — the ORB strategy has **no robust edge** (see below).

## ⚠️ Measured reality — accuracy is not profit

Directional accuracy alone means nothing without payoff and costs. A same-day,
cost-aware **A+ selective backtest** (STRONG + prime window + full trend
alignment, adaptive ATR stops, slippage, one entry per setup) on 28 Jul 2026:

| | BankNifty | Nifty50 |
|---|---|---|
| Distinct trades | 8 | 9 |
| Win rate | 25% | 44% |
| Profit factor | 0.20 | 0.52 |
| Net after cost+slippage | **−₹4,629** | **−₹3,300** |

**Why:** a typical 30-min move (~36 pts BankNifty / ~14 pts Nifty50) is barely
larger than round-trip cost+slippage (~27 / ~10 pts) — costs eat ~65–70% of the
move. Per-minute "hit-rates" of 50–72% are inflated ~8–11× by autocorrelation
(the same setup counted across many consecutive minutes); scored as **distinct
trades** the edge disappears. Improvements (adaptive stops, regime filter, real
metrics) improve *measurement*, not *edge*.

## Data / API status

- **Angel One SmartAPI** — primary real-time feed. Auto-login via TOTP (no SMS),
  works from abroad. Secrets in `.env` (see security note).
- **Yahoo Finance** — always-on fallback (can lag a few minutes).
- **Futures with real volume** (`fetch_futures.py`) — the index *spot* has volume=0
  (so its "VWAP" is fake); the **futures** carry real volume. Angel gives ~2.5yr
  daily and ~3mo of 5-min futures. Data rolls forward — re-run to append.
- **Options EOD with Open Interest** (`fetch_bhavcopy.py`) — NSE F&O bhavcopy is
  free and **not** geo-blocked (only the live option-chain API is). Gives **2 yr**
  of per-strike OHLC + volume + **OI + ΔOI** for both indices. IV is not included
  (compute from premium). *Intraday* per-strike OI is still paid-only.
- These unlock the OI / order-flow angle — see `experiments/` and the findings below.

## The honest truth (read this every week)

- Real, backtested accuracy is **~48–59%**, not 100%. Anyone selling 90%+ is a
  scammer. Every single one.
- **No strategy here has a proven robust edge** on the full backtest, and a
  cost-aware same-day test *lost money*. Free-data intraday index trading is
  roughly break-even before costs and negative after. Treat this as an
  **information / learning / paper-trading tool, not a way to make money.**
- The real frontier for genuine edge is order-flow / limit-order-book data
  (tick-level, colocated) — not free 5-minute candles. That is why retail cannot
  easily beat 55–60%: it is a data + speed gap, not an intelligence gap.
- Profitability = win-rate × **risk:reward** × sizing × discipline. A 40% win
  rate at 1:2.5 beats a 60% win rate at 1:0.8. Chase expectancy, not accuracy %.
- Expect losing streaks of 4–6 trades. At 1% risk that is a −4 to −6% dip. At
  10% risk it is account death. **Never raise the risk.**
- Rules that keep you alive:
  1. Respect NO-TRADE / WAIT. Sitting out is a signal too.
  2. Never hold past 15:15. No revenge trades.
  3. Paper-trade at least 20 sessions before real money.
  4. Only trade money you can afford to lose completely.

## Security

Secrets belong **only** in `.env` (git-ignored) — never in shared `.json` files.
The Angel TOTP secret is a live 2FA bypass; a `.gitignore` does **not** protect a
zipped folder. If `angel_config.json` was ever zipped/shared, **rotate the Angel
API key and reset TOTP.**

## What was tested and removed / archived (so nobody re-adds it blindly)

- **Legacy "Beast" engine** (VIX + cross-index extras) — head-to-head vs the main
  engine showed **no persistent edge** (it led in the morning, reversed by close;
  not better on the 90-day walk-forward). Archived to `experiments/legacy_engine.py`.
- **Volatility-regime engine** — the one thing that *held* on 90 days (~62–81%),
  but it predicts magnitude, **not a tradeable direction** — a component, not a
  trade. Archived to `experiments/volatility_engine.py` for future options work.
- **Meta-labeling** (triple-barrier, López de Prado) — tested twice, **collapsed
  out-of-sample**. Archived to `experiments/meta_labeling.py`.
- **ORB daily engine** — the earlier "positive-expectancy money engine" claim was
  a lucky short window; full 351-session backtest showed **no robust edge**
  (BankNifty −12.6R). `experiments/nifty_oracle.py`.
- **Rapid-fire 5-minute scalper** — lost in all 18 configs. Removed.
- **Chart / candlestick patterns as model inputs** — no improvement in two tests.
  Patterns stay UI-only.
- **Upstox integration** — never completed (user is abroad, needs SMS OTP);
  deleted. Angel One is the only live feed.

## The big investigation (100+ strategies) — what actually holds

Tested exhaustively on 17mo of price + 2yr of options-OI data, all per-day /
out-of-sample with first-half/second-half stability checks:

| Idea | Honest result |
|---|---|
| 50+ price signals (momentum, mean-rev, MA, RSI, Bollinger, gap, breakout…) | Nothing robustly > ~53–54% |
| Gap-fade (looked 62%) | Per-day = 45% — pure autocorrelation illusion |
| PCR-extremes (OI), next-day | 58% and briefly positive after cost — **then decayed and DIED** (last 6 mo: 36%, PF 0.74) |
| Cross-index lead-lag, money-flow, ensemble | ~51–54%, no breakout |
| **Volatility** prediction (magnitude, not direction) | ~54–65% per horizon — real, but it's *how big*, not *which way* |
| **VRP** (short ATM straddle) | 62–72% win — but a **Taleb distribution**: costs eat the edge and one bad week (−₹50–70k) can erase months. Retail cost ≈ the edge. |

**Bottom line:** no free-data configuration produces a robust, tradeable
directional edge after costs. This matches the academic literature (microstructure
noise dominates short horizons; the only real micro-edge is HFT order-book /
latency arbitrage, which needs colocated infrastructure). A high *win rate* is a
warning sign of hidden tail risk, not a sign of skill. Chase **expectancy**.
