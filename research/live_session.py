#!/usr/bin/env python3
"""The correction applied to a real session from the production log.

Everything else in this paper is simulated or backtested. This is the live
forecast stream that produced the headline number in the first place: the
same JSON the dashboard wrote while it was running, with the direction it
called each minute and the move that followed.

Two questions are answered here that a simulation cannot answer.

  1. What does the best session in the log actually support, once its interval
     is computed at an effective sample size rather than at the raw count of
     forecasts?

  2. Why is an estimator needed at all, when per-event scoring is free and
     already correctly calibrated? Three reasons, and they are visible here
     rather than argued: a live dashboard must print a number every minute and
     cannot wait k bars; subsampling discards k-1 of every k observations and so
     produces a wider interval than necessary; and WHICH of the k subsamples is
     chosen changes the answer, sometimes by a lot.

Run:  python3 -m research.live_session
"""
import argparse
import json

import numpy as np

from oracle import paths
from research import common as C
from research.effective_sample import (naive_interval, corrected_interval,
                                       block_bootstrap_interval,
                                       run_corrected_interval)

LOG = paths.RUNTIME / "forecast_history_v2.json"


def sessions_from_log(horizon):
    """Per-session hit sequences, in the order the dashboard emitted them."""
    if not LOG.exists() or LOG.stat().st_size == 0:
        raise FileNotFoundError(f"{LOG} missing — run the dashboard to populate it")
    log = json.loads(LOG.read_text())

    out = {}
    for index, rows in log.items():
        by_day = {}
        for r in rows:
            h = r.get("h", {}).get(horizon)
            if not h or not h.get("dir") or h.get("move") is None:
                continue
            move = h["move"]
            if move == 0:
                continue
            hit = int((move > 0) == (h["dir"] == "UP"))
            by_day.setdefault(r.get("day", r.get("time", "")[:10] or "session"),
                              []).append(hit)
        out[index] = {d: np.array(v) for d, v in by_day.items() if len(v) >= 40}
    return out


def subsample_spread(h, k):
    """The k possible per-event subsamples of one session.

    Per-event scoring is unbiased and correctly calibrated, but it requires
    choosing an offset, and the choice is arbitrary. The spread across offsets
    is how much that arbitrary choice is worth.
    """
    return np.array([h[o::k].mean() for o in range(k) if len(h[o::k]) >= 5])


def report(index, day, h, k):
    n = len(h)
    p, (nl, nh), _ = naive_interval(h)
    _, (hl, hh), n_eff_hac, _ = corrected_interval(h)
    _, (bl, bh) = block_bootstrap_interval(h, rng=np.random.default_rng(C.SEED))
    _, (rl, rh), n_eff_run, _ = run_corrected_interval(h)
    subs = subsample_spread(h, k)

    print(f"\n  {index}  {day}   {n} forecasts, hit-rate {p*100:.1f}%")
    print(f"    {'estimator':<26}{'95% interval':>20}{'width':>9}{'n_eff':>8}")
    print("    " + "-" * 63)
    for lbl, (lo, hi), ne in (("naive binomial at n", (nl, nh), n),
                              ("HAC (Bartlett)", (hl, hh), n_eff_hac),
                              ("block bootstrap", (bl, bh), None),
                              ("run-structure design effect", (rl, rh), n_eff_run)):
        print(f"    {lbl:<26}{f'[{lo*100:.1f}, {hi*100:.1f}]':>20}"
              f"{(hi-lo)*100:>9.1f}{(f'{ne:,.0f}' if ne else '—'):>8}")
    print(f"    {'per-event, k subsamples':<26}"
          f"{f'{subs.min()*100:.1f} to {subs.max()*100:.1f}':>20}"
          f"{(subs.max()-subs.min())*100:>9.1f}{len(h)//k:>8}")

    excl = rl > 0.5
    print(f"    -> at n_eff the interval {'EXCLUDES' if excl else 'includes'} 50%"
          f"; the session {'supports' if excl else 'does not support'} a claim of skill")
    return {"index": index, "day": str(day), "n": int(n), "hit_rate": round(p, 4),
            "naive_ci": [round(nl, 4), round(nh, 4)],
            "hac_ci": [round(hl, 4), round(hh, 4)],
            "bootstrap_ci": [round(bl, 4), round(bh, 4)],
            "run_ci": [round(rl, 4), round(rh, 4)],
            "n_eff_run": round(n_eff_run, 1),
            "subsample_min": round(float(subs.min()), 4),
            "subsample_max": round(float(subs.max()), 4),
            "subsample_spread_pts": round(float(subs.max() - subs.min()) * 100, 1),
            "excludes_half_at_n_eff": bool(excl)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="30m", choices=list(C.HORIZON_BARS))
    ap.add_argument("--top", type=int, default=2, help="best sessions per index")
    a = ap.parse_args()
    k = C.HORIZON_BARS[a.horizon]

    print(f"\n{'='*78}")
    print(f"THE CORRECTION ON A REAL SESSION — production log, {a.horizon} horizon")
    print(f"{'='*78}")
    print("  Not simulated. This is the forecast stream the dashboard actually")
    print("  wrote, scored against the moves that actually followed.")

    data = sessions_from_log(a.horizon)
    out = []
    for index, days in data.items():
        best = sorted(days.items(), key=lambda kv: -kv[1].mean())[:a.top]
        for day, h in best:
            out.append(report(index, day, h, k))

    spreads = [o["subsample_spread_pts"] for o in out]
    print(f"\n{'='*78}\nWHY NOT SIMPLY SCORE PER EVENT\n{'='*78}")
    print(f"  Per-event scoring is unbiased and correctly calibrated, and it is")
    print(f"  what we use for every backtest in this paper. It is not sufficient")
    print(f"  for a live stream, for three reasons visible above.")
    print(f"\n  1. A dashboard must print a number every minute. It cannot wait")
    print(f"     {k} bars, and it cannot discard {k-1} of every {k} forecasts it has")
    print(f"     already made.")
    print(f"  2. Subsampling throws away {(1-1/k)*100:.0f}% of the observations, so its")
    print(f"     honest interval is wider than it needs to be. The corrected")
    print(f"     estimator uses the whole stream.")
    print(f"  3. WHICH subsample is arbitrary. Across the {k} possible offsets the")
    print(f"     same session reports hit-rates spanning "
          f"{min(spreads):.0f} to {max(spreads):.0f} points.")
    print(f"     An analyst free to pick the offset is free to pick the answer.")

    d = paths.ROOT / "results"; d.mkdir(exist_ok=True)
    f = d / f"live_session_{a.horizon}.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {f.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
