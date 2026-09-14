#!/usr/bin/env python3
"""
Pre-train / retrain the forecast models (both indices) without launching
the dashboard. Useful after new data is fetched or the engine changes.

The models auto-train on the first dashboard launch too — this is just a
convenience so the first launch is instant.

Usage:  python3 train_models.py            # trains & caches BankNifty + Nifty50
        python3 train_models.py --fresh    # delete caches first, force retrain
"""
import argparse
from pathlib import Path
from oracle.models import direction as fce
from oracle import paths

CACHE = paths.CACHE
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true",
                    help="delete cached models and retrain from scratch")
    args = ap.parse_args()

    if args.fresh:
        for f in CACHE.glob("*_models_v2.pkl"):
            f.unlink()
            print(f"deleted {f.name}")

    for name in ["BANKNIFTY", "NIFTY50"]:
        print(f"\n--- {name} ---")
        eng = fce.ForecastEngine(name, 100000, 1.0)
        print(f"  out-of-sample: {eng.backtest_acc}")
        print(f"  STRONG signal: ~{eng.strong_acc}% on {eng.strong_n} val minutes")

    print("\n✅ Models trained and cached. Launch: python3 nifty_dashboard.py")


if __name__ == "__main__":
    main()
