#!/usr/bin/env python3
"""What accuracy would actually be needed, and what accuracy is achievable.

Directional accuracy is the number everyone quotes. It is also, on its own,
meaningless -- because whether a given accuracy makes money depends entirely on
how large the move is relative to the cost of capturing it.

This script measures both sides of that comparison on real data:

  ACHIEVABLE   the best out-of-sample accuracy any model family reached
               (from research/model_zoo.py)

  REQUIRED     the accuracy a symmetric-payoff strategy needs to break even,
               derived from the measured distribution of k-bar moves and a
               stated round-trip cost

For a symmetric bracket with target and stop both equal to the typical move m,
and round-trip cost c, expectancy per trade is

    E = p(m - c) - (1 - p)(m + c) = m(2p - 1) - c

so breaking even requires

    p* = 1/2 + c / (2m)

The ratio c/m is therefore the entire game. When it is small, a 52% edge is a
business. When it approaches 1, no achievable accuracy is enough -- and this is
the regime NSE index intraday trading sits in.

Run:  python3 -m research.cost_geometry [--cost-bp 4]
"""
import argparse
import json

import numpy as np

from oracle import paths
from research import common as C

# Round-trip cost in basis points of notional. 4 bp is a realistic futures
# figure (brokerage + STT + exchange + stamp + slippage). Options are worse:
# spread plus theta over a 30-minute hold routinely runs several times this.
DEFAULT_COST_BP = 4.0


def move_distribution(name, horizon):
    """Absolute k-bar move, in points and in basis points of price."""
    k = C.HORIZON_BARS[horizon]
    close = C.load_bars(name)["Close"]
    day = close.index.normalize()

    fwd_pts = (close.shift(-k) - close)
    same = day.values == np.roll(day.values, -k)
    fwd_pts = fwd_pts[same & fwd_pts.notna()]
    fwd_bp = (fwd_pts / close.reindex(fwd_pts.index)) * 1e4

    a_pts, a_bp = fwd_pts.abs(), fwd_bp.abs()
    return {
        "price": float(close.mean()),
        "median_move_pts": float(a_pts.median()),
        "mean_move_pts": float(a_pts.mean()),
        "median_move_bp": float(a_bp.median()),
        "p25_bp": float(a_bp.quantile(0.25)),
        "p75_bp": float(a_bp.quantile(0.75)),
        "n": int(len(a_pts)),
        "_abs_bp": a_bp,
    }


def breakeven(cost_bp, move_bp):
    """Accuracy needed for zero expectancy at a symmetric payoff."""
    if move_bp <= 0:
        return float("nan")
    return 0.5 + cost_bp / (2 * move_bp)


def run(name, cost_bp, achieved):
    print(f"\n{'='*80}\n{name} — cost geometry\n{'='*80}")
    print(f"  round-trip cost assumed: {cost_bp:.1f} bp")
    print(f"\n{'horizon':<10}{'median move':>13}{'cost/move':>11}"
          f"{'break-even acc':>16}{'move>cost':>11}{'achieved':>10}")
    print("-" * 80)

    rows = []
    for horizon in C.HORIZON_BARS:
        d = move_distribution(name, horizon)
        m_bp = d["median_move_bp"]
        ratio = cost_bp / m_bp
        p_star = breakeven(cost_bp, m_bp)
        clears = float((d["_abs_bp"] > cost_bp).mean())
        ach = achieved.get(horizon)

        print(f"{horizon:<10}{d['median_move_pts']:>8.1f} pts"
              f"{ratio:>11.2f}{p_star*100:>15.1f}%{clears*100:>10.0f}%"
              f"{(f'{ach*100:.1f}%' if ach else '—'):>10}")

        rows.append({"horizon": horizon, "median_move_pts": round(d["median_move_pts"], 1),
                     "median_move_bp": round(m_bp, 2),
                     "cost_over_move": round(ratio, 3),
                     "breakeven_accuracy": round(p_star, 4),
                     "frac_bars_move_exceeds_cost": round(clears, 4),
                     "achieved_accuracy": ach, "n": d["n"]})

    p30 = next(r for r in rows if r["horizon"] == "30m")
    ach30 = p30["achieved_accuracy"]
    print(f"\n  At the 30-minute horizon this index needs "
          f"{p30['breakeven_accuracy']*100:.0f}% to break even.")
    if ach30:
        gap = (p30["breakeven_accuracy"] - ach30) * 100
        print(f"  The best model family achieved {ach30*100:.1f}% — short by "
              f"{gap:.0f} percentage points.")
    print(f"  Even a genuine 85% would clear break-even by only "
          f"{(0.85 - p30['breakeven_accuracy'])*100:+.0f} points here.")

    return {"index": name, "cost_bp": cost_bp, "horizons": rows}


def sensitivity(name, horizon="30m"):
    """Break-even accuracy across plausible cost regimes."""
    d = move_distribution(name, horizon)
    m_bp = d["median_move_bp"]
    print(f"\n  {name} {horizon}: median move {m_bp:.1f} bp "
          f"({d['median_move_pts']:.0f} pts)")
    print(f"    {'cost (bp)':<12}{'instrument':<34}{'break-even accuracy':>20}")
    print("    " + "-" * 66)
    out = []
    for cost, note in ((1.0, "exchange fees only, no slippage"),
                       (2.0, "tight futures, minimal slippage"),
                       (4.0, "realistic futures round trip"),
                       (8.0, "futures with adverse slippage"),
                       (20.0, "ATM options: spread + 30 min of theta"),
                       (40.0, "OTM options or a wide book")):
        p = breakeven(cost, m_bp)
        shown = f"{p*100:.1f}%" if p < 1 else "impossible"
        print(f"    {cost:<12.0f}{note:<34}{shown:>20}")
        out.append({"cost_bp": cost, "instrument": note,
                    "breakeven_accuracy": (round(p, 4) if p < 1 else None)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-bp", type=float, default=DEFAULT_COST_BP)
    ap.add_argument("--indices", nargs="*", default=C.INDICES)
    a = ap.parse_args()

    # Best achieved accuracy per horizon, read from the model zoo if it has run.
    achieved = {}
    for f in sorted((paths.ROOT / "results").glob("model_zoo_*.json")):
        for blk in json.loads(f.read_text()):
            best = max((m["accuracy"] for m in blk["models"]
                        if m["model"] != "majority-class baseline"), default=None)
            achieved.setdefault(blk["index"], {})[blk["horizon"]] = best

    out = [run(n, a.cost_bp, achieved.get(n, {})) for n in a.indices]

    print(f"\n{'='*80}\nSENSITIVITY — break-even accuracy by cost regime\n{'='*80}")
    for n in a.indices:
        s = sensitivity(n)
        next(o for o in out if o["index"] == n)["sensitivity"] = s

    print(f"\n{'='*80}\nVERDICT\n{'='*80}")
    print("  The binding constraint is not model accuracy. It is the ratio of the")
    print("  typical move to the cost of capturing it. On these instruments that")
    print("  ratio leaves a break-even requirement far above anything any model")
    print("  family reached — and above the inflated numbers this field advertises.")
    print("\n  Statistical significance and economic significance are different")
    print("  questions. A 51.3% edge can be real at p=0.001 and still lose money")
    print("  on every trade.")

    d = paths.ROOT / "results"; d.mkdir(exist_ok=True)
    f = d / "cost_geometry.json"
    f.write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {f.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
