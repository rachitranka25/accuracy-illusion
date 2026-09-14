# Pulse + Forecast Window Upgrade - 2026-08-05

## Goal

Upgrade the Pulse window and Forecast window to behave more like a high-selectivity
decision system: fewer weak calls, clearer best-setup guidance, faster green
ticks where the market allows them, and honest scoring that does not fake 90%
accuracy.

## Files Changed

- `nifty_dashboard.py`
- `forecast_feed.html`
- `pulse_view.html`

## Backend Changes

### Elite forecast selector

Added `_forecast_quality_pick(rec)` in `nifty_dashboard.py`.

It chooses the best actionable forecast from the current multi-horizon model
output using only causal information:

- calibrated tier accuracy
- model confidence
- direction alignment
- STRONG / ULTRA flags
- high-volatility conviction flag
- prime vs midday window
- fast-horizon preference

The selector intentionally focuses on `5m` and `15m` horizons for the elite
track because the saved history shows longer horizons are slower and noisier for
the "minimum time" goal.

### Elite stats API payload

Added `_elite_forecast_stats(rows)` and extended `/api/forecasts` with:

- `best_now`: current SNIPE / TAKE / WAIT recommendation
- `elite`: elite-only green-rate stats and recent elite rows

This lets the Forecast window separate noisy every-minute forecasts from the
small set of best setups.

### Pulse Sniper mode

Added `sniper` flavor to `/api/pulse`:

- tighter target geometry
- stricter chop filter
- stricter speed gate
- stricter momentum gate
- requires STRONG prime-hour setup plus ULTRA / high-conviction / high quality
- longer re-entry gap to avoid stacking mediocre trades

### Pulse quality score

Each Pulse call now includes `quality`, built from:

- alignment count
- 30m confidence
- STRONG / ULTRA / high-conviction flags
- midday penalty

This score is displayed in the Pulse history so weak or below-bar calls are
visibly different from clean setups.

### Green-rate accounting

Pulse headline scoring now uses:

- `HIT` = full target hit
- `PROFIT` = partial green exit
- `MISS` = stop hit or losing exit

The headline `pct` is now green rate:

`(HIT + PROFIT) / resolved calls`

The API also returns:

- `hit_pct`
- `greens`
- `net_pts`
- `avg_green_min`

## Forecast Window UI Changes

Added an elite setup strip above each index card:

- SNIPE / TAKE / WAIT status
- direction and horizon
- grade and quality score
- target and stop
- expected points
- points-per-minute speed
- elite green-rate today
- ULTRA and HIGH VOL tags when present

The detailed history table still exists, but elite rows are tagged so the best
setups stand out from the normal forecast stream.

## Pulse Window UI Changes

Added the `Sniper` flavor as the default view.

Updated the display to show:

- green rate instead of target-hit-only rate
- full hits vs partial greens vs misses
- net points today
- average green time
- setup quality score on rows
- ULTRA marker on the current signal

## Validation

Ran Python source compilation in memory:

```bash
python3 -c "import pathlib; [compile(pathlib.Path(f).read_text(), f, 'exec') for f in ['nifty_dashboard.py','forecast_engine.py','meta_filter.py']] ; print('python source compile OK')"
```

Result:

```text
python source compile OK
```

The system `/usr/bin/python3` does not have `pandas`, so import validation was
run with the project-compatible Anaconda Python:

```bash
/opt/anaconda3/bin/python3.13 -c "import nifty_dashboard as nd; print('import OK', bool(nd._forecast_quality_pick({'h':{}}) is None))"
```

Result:

```text
import OK True
```

Recent saved-history elite snapshot after tightening to fast horizons:

```text
BANKNIFTY 52% green, 114/221 elite resolved, avg score 84.4, avg speed 8.99 pts/min
NIFTY50   51% green, 111/216 elite resolved, avg score 86.7, avg speed 2.48 pts/min
FINNIFTY  64% green,  61/96  elite resolved, avg score 87.2, avg speed 5.52 pts/min
```

## Important Reality Check

This upgrade does not promise 90% trading accuracy. It makes the windows more
selective and more useful by suppressing weaker conditions, emphasizing fast
setups, and separating full hits from partial green exits. Real intraday markets
remain noisy; the edge comes from selectivity, speed, risk control, and sitting
out bad windows.
