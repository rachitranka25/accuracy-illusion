#!/usr/bin/env python3
"""How a forecaster with no edge produces a 71% day, and an 85% day.

This is the mechanism the whole study turns on, isolated and measured.

Two facts combine, and neither is enough on its own.

  (a) OVERLAPPING OUTCOMES. A k-bar-ahead forecast emitted every bar shares k-1
      of its k outcome bars with its neighbour. The truth series is autocorrelated
      by construction -- measured at 0.84 at lag 1, decaying to zero at exactly
      lag k (see research/ceiling_test.py).

  (b) PERSISTENT PREDICTIONS. A real model does not re-roll every bar. Its
      features move slowly, so it repeats the same call for long stretches.
      Measured on this project's own live forecast log, the 30-minute engine
      holds a direction for a mean of 15 minutes and as long as 135.

With independent predictions, overlap alone changes nothing -- the hit indicator
is still i.i.d. It is the combination that matters: when both the call and the
outcome persist, one lucky trending stretch is recorded as dozens of hits, and
the reported hit-rate acquires a variance far larger than its sample size
implies.

The consequence is that a coinflip does not sit in a tight band around 50%. It
produces a wide distribution whose upper tail reaches the numbers that get
published, marketed and believed.

Experiments:

  0. PERSISTENCE     Measure the real engine's run-length distribution from the
                     live forecast log, so the simulation is calibrated to this
                     system's actual behaviour rather than an assumption.

  1. REAL DATA       A zero-skill forecaster with that persistence, on actual
                     BANKNIFTY and NIFTY 50 bars, scored the way a live dashboard
                     scores it. How often does it post a 60 / 71 / 85% session?

  2. CALIBRATION     Realised spread against the confidence interval the reported
                     sample size implies. The ratio is how badly the stated
                     interval understates the truth.

  3. RECOVERY        The identical forecasts scored one observation per k bars.
                     If the stated and realised intervals agree there, the
                     inflation was the scoring rule and nothing else.

Run:  python3 -m research.overlap_simulation [--trials 200]
"""
import argparse
import json

import numpy as np

from oracle import paths
from research import common as C

LOG = paths.RUNTIME / "forecast_history_v2.json"


# ── 0. how persistent is the real engine? ──────────────────────────────
def measured_run_lengths(horizon):
    """Run lengths of identical consecutive calls, from the live forecast log.

    Returns run lengths in *bars* (the log is per-minute; bars are 5 minutes),
    or None when the log is unavailable.
    """
    if not LOG.exists() or LOG.stat().st_size == 0:
        return None, {}
    try:
        log = json.loads(LOG.read_text())
    except Exception:
        return None, {}

    per_index, pooled = {}, []
    for index, rows in log.items():
        seq = [r["h"][horizon]["dir"] for r in rows
               if horizon in r.get("h", {}) and r["h"][horizon].get("dir")]
        if len(seq) < 50:
            continue
        runs, cur = [], 1
        for a, b in zip(seq, seq[1:]):
            if a == b:
                cur += 1
            else:
                runs.append(cur); cur = 1
        runs.append(cur)
        per_index[index] = {"forecasts": len(seq), "runs": len(runs),
                            "mean_run_min": round(float(np.mean(runs)), 1),
                            "median_run_min": int(np.median(runs)),
                            "max_run_min": int(max(runs))}
        pooled += runs

    if not pooled:
        return None, {}
    bars = np.maximum(1, np.round(np.array(pooled) / 5)).astype(int)
    return bars, per_index


# ── truth from real bars ───────────────────────────────────────────────
def session_truth(name, k):
    """Per-session arrays of the real k-bar-ahead direction."""
    close = C.load_bars(name)["Close"]
    fwd = close.shift(-k) / close - 1
    day = close.index.normalize()
    same = day.values == np.roll(day.values, -k)
    truth = (fwd > 0).astype(float).where(fwd.notna() & same)

    out = []
    for _, g in truth.groupby(day):
        g = g.dropna()
        if len(g) >= 40:
            out.append(g.values.astype(int))
    return out


def simulate(sessions, run_bars, rng, trials, stride=1):
    """Score a zero-skill persistent forecaster. stride=k gives non-overlapping."""
    rates, sizes = [], []
    for t in sessions:
        for _ in range(trials):
            pred = C.persistent_forecast(t, run_bars, rng, bias=0.5)
            idx = np.arange(0, len(t), stride)
            rates.append((pred[idx] == t[idx]).mean())
            sizes.append(len(idx))
    return np.array(rates), float(np.mean(sizes))


def report(arr, n_reported, label):
    """Realised spread versus the interval the reported n implies."""
    stated_sd = np.sqrt(0.25 / n_reported)
    realised_sd = arr.std()
    factor = realised_sd / stated_sd

    print(f"\n  {label}")
    print(f"    reported sample size per session : {n_reported:.0f} forecasts")
    print(f"    stated binomial sd               : {stated_sd*100:5.2f} pts "
          f"(95% CI +/-{1.96*stated_sd*100:.1f})")
    print(f"    realised sd across sessions      : {realised_sd*100:5.2f} pts "
          f"(95% band +/-{1.96*realised_sd*100:.1f})")
    print(f"    UNDERSTATEMENT FACTOR            : {factor:5.2f}x")
    print(f"    mean {arr.mean()*100:.1f}%   p95 {np.percentile(arr,95)*100:.1f}%   "
          f"p99 {np.percentile(arr,99)*100:.1f}%   max {arr.max()*100:.1f}%")

    tails = {}
    for thr in (0.60, 0.71, 0.80, 0.85):
        frac = float((arr >= thr).mean())
        tails[f"p_session_ge_{int(thr*100)}"] = round(frac, 5)
        once = f"1 in {1/frac:,.0f} sessions" if frac > 0 else "not observed"
        print(f"    P(session >= {int(thr*100)}%) = {frac*100:6.2f}%   ({once})")

    # Keep the empirical distribution itself, so figures plot what was measured
    # rather than a normal curve fitted to its first two moments.
    counts, edges = np.histogram(arr, bins=60, range=(0.0, 1.0))
    histogram = {"bin_edges": [round(float(e), 4) for e in edges],
                 "counts": [int(c) for c in counts]}

    return {"reported_n": round(n_reported, 1),
            "stated_sd": round(float(stated_sd), 4),
            "realised_sd": round(float(realised_sd), 4),
            "understatement_factor": round(float(factor), 3),
            "mean": round(float(arr.mean()), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
            "p99": round(float(np.percentile(arr, 99)), 4),
            "max": round(float(arr.max()), 4),
            "simulated_sessions": int(len(arr)),
            "histogram": histogram, **tails}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="30m", choices=list(C.HORIZON_BARS))
    ap.add_argument("--trials", type=int, default=200)
    a = ap.parse_args()

    k = C.HORIZON_BARS[a.horizon]
    rng = np.random.default_rng(C.SEED)
    out = {"horizon": a.horizon, "horizon_bars": k}

    # ── 0. persistence ─────────────────────────────────────────────────
    print(f"\n{'='*78}\n0. PERSISTENCE — how long the real engine repeats a call\n{'='*78}")
    run_bars, per_index = measured_run_lengths(a.horizon)
    if run_bars is None:
        run_bars = np.maximum(1, rng.geometric(1 / 3, 5000))
        print("   live log unavailable — falling back to geometric(mean 3 bars)")
        out["persistence_source"] = "fallback geometric"
    else:
        for idx, st in per_index.items():
            print(f"   {idx:<10} {st['forecasts']:>5} forecasts, {st['runs']:>4} runs, "
                  f"mean run {st['mean_run_min']:>5.1f} min, max {st['max_run_min']} min")
        print(f"\n   pooled: mean {run_bars.mean():.1f} bars "
              f"({run_bars.mean()*5:.0f} min), max {run_bars.max()} bars")
        print("   A model that re-rolled every bar would have a mean run of 1.")
        out["persistence_source"] = "measured from live forecast log"
        out["persistence_per_index"] = per_index
    out["mean_run_bars"] = round(float(run_bars.mean()), 2)

    # ── 1 & 2. per-bar scoring on real data ────────────────────────────
    print(f"\n{'='*78}\n1. ZERO-SKILL FORECASTER ON REAL BARS — scored every bar\n{'='*78}")
    print("   No model, no features. Fair coin, held for a realistic run length.")
    for name in C.INDICES:
        sessions = session_truth(name, k)
        arr, n_rep = simulate(sessions, run_bars, rng, a.trials, stride=1)
        out[f"per_bar_{name}"] = report(arr, n_rep, f"{name} — scored every bar")

    # ── 3. recovery ────────────────────────────────────────────────────
    print(f"\n{'='*78}\n3. RECOVERY — identical forecaster, scored once per k bars\n{'='*78}")
    for name in C.INDICES:
        sessions = session_truth(name, k)
        arr, n_rep = simulate(sessions, run_bars, rng, a.trials, stride=k)
        out[f"non_overlap_{name}"] = report(arr, n_rep, f"{name} — one obs per {k} bars")

    # ── verdict ────────────────────────────────────────────────────────
    bn = out[f"per_bar_{C.INDICES[0]}"]
    nv = out[f"non_overlap_{C.INDICES[0]}"]
    print(f"\n{'='*78}\nVERDICT\n{'='*78}")
    print(f"  Scored every bar, a forecaster with exactly zero skill:")
    print(f"    reaches >=60% on {bn['p_session_ge_60']*100:.1f}% of sessions")
    print(f"    reaches >=71% on {bn['p_session_ge_71']*100:.2f}% of sessions")
    print(f"    reaches >=85% on {bn['p_session_ge_85']*100:.2f}% of sessions")
    print(f"  and its stated confidence interval is {bn['understatement_factor']:.1f}x too narrow.")
    print(f"\n  Scored once per {k} bars, the same forecaster's stated interval is")
    print(f"  {nv['understatement_factor']:.2f}x the realised spread — correctly calibrated.")

    # Multiplicity: a dashboard does not watch one number, it watches a grid.
    n_series = len(C.INDICES) * len(C.HORIZON_BARS)
    p71 = bn["p_session_ge_71"]
    any71 = 1 - (1 - p71) ** n_series
    print(f"\n  MULTIPLICITY. A dashboard does not watch one number. This one shows")
    print(f"  {len(C.INDICES)} indices x {len(C.HORIZON_BARS)} horizons = {n_series} hit-rates every session.")
    print(f"    P(at least one >=71% today) = {any71*100:.1f}%  "
          f"-> about once every {1/any71:.0f} sessions")
    print(f"  So a headline 71% arrives in the first fortnight of live operation")
    print(f"  with no edge whatsoever. This project's own 71% was exactly this:")
    print(f"  one index, one horizon, one trending Friday.")
    out["multiplicity"] = {"series_watched": n_series,
                           "p_any_series_ge_71": round(float(any71), 4),
                           "sessions_between": round(float(1 / any71), 1)}

    print(f"\n  The difference between a discovery and an artefact here is not the")
    print(f"  model. It is the denominator.")

    d = paths.ROOT / "results"; d.mkdir(exist_ok=True)
    f = d / f"overlap_simulation_{a.horizon}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {f.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
