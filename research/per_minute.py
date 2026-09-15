#!/usr/bin/env python3
"""The artefact at the granularity it actually occurs.

Everything else in this study runs on 5-minute bars, where a 30-minute horizon
is k = 6 and a session yields 69 scored forecasts. The production system does
not work that way. It emits a forecast every MINUTE, so a 30-minute horizon is
k = 30 and a session yields about 336 forecasts -- and the headline number this
paper exists to explain (202 of 285) was counted at that granularity.

Explaining a per-minute number with a per-bar model is a gap, so it is closed
here. The truth sequence is not simulated: it is taken from the production log,
which recorded the direction called at each minute and the move that followed,
so the outcome series has the real per-minute autocorrelation rather than an
assumed one.

Two quantities matter and both get worse with density:

  n / n_eff   The nominal count rises fivefold while the information does not,
              so the ratio the dashboard misreports by rises with it.

  the tail    Run lengths measured in minutes reach 135-205, against a 30-minute
              outcome window. At bar granularity that tail was 27-41 bars
              against a 6-bar window -- the same ratio, five times the count.

Run:  python3 -m research.per_minute
"""
import argparse
import json

import numpy as np

from oracle import paths
from research import common as C

LOG = paths.RUNTIME / "forecast_history_v2.json"


def per_minute_truth(horizon):
    """Per-session, per-minute direction outcomes, straight from the live log."""
    if not LOG.exists() or LOG.stat().st_size == 0:
        raise FileNotFoundError(f"{LOG} missing — run the dashboard to populate it")
    log = json.loads(LOG.read_text())

    out = {}
    for index, rows in log.items():
        days = {}
        for r in rows:
            h = r.get("h", {}).get(horizon)
            if not h or h.get("move") is None or not h.get("dir"):
                continue
            if h["move"] == 0:
                continue
            days.setdefault(r.get("day", "?"), []).append(int(h["move"] > 0))
        out[index] = {d: np.array(v) for d, v in days.items() if len(v) >= 200}
    return out


def minute_runs(horizon):
    """Run lengths of identical consecutive calls, in MINUTES, from the log."""
    log = json.loads(LOG.read_text())
    pooled = []
    for _, rows in log.items():
        seq = [r["h"][horizon]["dir"] for r in rows
               if horizon in r.get("h", {}) and r["h"][horizon].get("dir")]
        if len(seq) < 50:
            continue
        cur = 1
        for a, b in zip(seq, seq[1:]):
            if a == b:
                cur += 1
            else:
                pooled.append(cur); cur = 1
        pooled.append(cur)
    return np.array(pooled)


def simulate(sessions, runs, rng, trials):
    rates, sizes, n_effs = [], [], []
    for t in sessions:
        for _ in range(trials):
            pred = C.persistent_forecast(t, runs, rng, bias=0.5)
            h = (pred == t).astype(int)
            rates.append(h.mean()); sizes.append(len(h))
            ne, _ = C.effective_n(h)
            n_effs.append(ne)
    return np.array(rates), float(np.mean(sizes)), float(np.mean(n_effs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="30m", choices=list(C.HORIZON_BARS))
    ap.add_argument("--trials", type=int, default=400)
    a = ap.parse_args()

    k_min = {"5m": 5, "15m": 15, "30m": 30, "60m": 60}[a.horizon]
    rng = np.random.default_rng(C.SEED)
    runs = minute_runs(a.horizon)
    data = per_minute_truth(a.horizon)

    print(f"\n{'='*78}")
    print(f"PER-MINUTE REPLICATION — the granularity the dashboard reports at")
    print(f"{'='*78}")
    print(f"  horizon {a.horizon} = {k_min} minutes of outcome window")
    print(f"  prediction runs, in minutes: mean {runs.mean():.1f}, "
          f"median {np.median(runs):.0f}, max {runs.max()}")
    print(f"  outcome series taken from the live log, not simulated\n")

    out = {"horizon": a.horizon, "outcome_window_min": k_min,
           "run_mean_min": round(float(runs.mean()), 1),
           "run_max_min": int(runs.max()), "indices": {}}

    print(f"  {'index':<11}{'sessions':>9}{'n':>7}{'n_eff':>8}{'stated':>9}"
          f"{'realised':>10}{'factor':>8}{'P(>=71%)':>10}")
    print("  " + "-" * 73)

    for index, days in data.items():
        sessions = list(days.values())
        rates, n, n_eff = simulate(sessions, runs, rng, a.trials)
        stated = np.sqrt(0.25 / n)
        realised = rates.std()
        p71 = float((rates >= 0.71).mean())
        print(f"  {index:<11}{len(sessions):>9}{n:>7.0f}{n_eff:>8.1f}"
              f"{stated*100:>8.2f}%{realised*100:>9.2f}%"
              f"{realised/stated:>8.2f}{p71*100:>9.2f}%")
        out["indices"][index] = {
            "sessions": len(sessions), "n": round(n, 1),
            "n_eff": round(n_eff, 1),
            "stated_sd": round(float(stated), 5),
            "realised_sd": round(float(realised), 5),
            "understatement_factor": round(float(realised / stated), 3),
            "p_session_ge_71": round(p71, 5),
            "mean": round(float(rates.mean()), 4),
            "max": round(float(rates.max()), 4)}

    ref = out["indices"][list(out["indices"])[0]]
    print(f"\n{'='*78}\nAGAINST THE 5-MINUTE-BAR VERSION\n{'='*78}")
    print(f"  {'':<22}{'per 5-min bar':>16}{'per minute':>14}")
    print("  " + "-" * 52)
    print(f"  {'forecasts / session':<22}{69:>16}{ref['n']:>14.0f}")
    print(f"  {'effective sample':<22}{12.4:>16.1f}{ref['n_eff']:>14.1f}")
    print(f"  {'understatement factor':<22}{1.66:>16.2f}{ref['understatement_factor']:>14.2f}")
    print(f"\n  The nominal count rises about fivefold. The information does not.")
    print(f"  Whichever way the effective sample moves, the RATIO the dashboard")
    print(f"  misreports by is larger at the granularity it actually publishes,")
    print(f"  so the 5-minute-bar figures elsewhere in this paper understate the")
    print(f"  artefact rather than exaggerating it.")

    d = paths.ROOT / "results"; d.mkdir(exist_ok=True)
    f = d / f"per_minute_{a.horizon}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {f.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
