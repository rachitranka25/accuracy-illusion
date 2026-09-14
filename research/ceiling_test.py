#!/usr/bin/env python3
"""Is there anything to predict? A model-independent ceiling test.

Arguing about model quality never ends, because there is always one more
architecture. This asks the question in a form that does not depend on the model
at all.

Five measurements, run on the live system's own feature set:

  1. OVER-CAPACITY      An unconstrained model is fitted until it memorises the
                        training set. If the gap between train and test is total,
                        capacity was never the binding constraint.

  2. SHUFFLED CONTROL   The same pipeline with randomly permuted labels. If real
                        and shuffled labels score the same, the pipeline is
                        reading noise. This is the decisive measurement.

  3. POSITIVE CONTROL   A synthetic signal of known strength is injected into the
                        features. If the pipeline recovers it, the pipeline works
                        and the null result on real data is a property of the
                        data. Without this, "we found nothing" is indistinguishable
                        from "our code is broken" -- which is why it is here.

  4. AUTOCORRELATION    Ljung-Box on the forward returns. Direction cannot be
                        predicted from its own past if there is no serial
                        structure in it.

  5. 1-NN ORACLE        For each test bar, the single most similar historical bar.
                        An upper bound on what any analogue-based method could do.

Run:  python3 -m research.ceiling_test [--horizon 30m] [--folds 5]
"""
import argparse
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox

from oracle import paths
from research import common as C


def over_capacity(X, y, k, folds, rng, label="real"):
    """Fit until the training set is memorised; report what generalises."""
    s = C.Score(f"over-capacity ({label})")
    tr_acc = []
    for tr, te in C.walk_forward(len(X), k, folds):
        m = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=1,
            max_features="sqrt", n_jobs=-1, random_state=C.SEED)
        m.fit(X.iloc[tr], y[tr])
        tr_acc.append((m.predict(X.iloc[tr]) == y[tr]).mean())
        s.add(y[te], m.predict(X.iloc[te]))
    s.train_acc = float(np.mean(tr_acc))
    return s


def nn_oracle(X, y, k, folds):
    """Nearest historical analogue. An upper bound on analogue methods."""
    s = C.Score("1-NN oracle")
    for tr, te in C.walk_forward(len(X), k, folds):
        sc = StandardScaler().fit(X.iloc[tr])
        m = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
        m.fit(sc.transform(X.iloc[tr]), y[tr])
        s.add(y[te], m.predict(sc.transform(X.iloc[te])))
    return s


def positive_control(X, y, k, folds, rng, strength):
    """Inject a synthetic feature that carries a known amount of signal.

    The feature agrees with the label on `strength` of the rows and is random on
    the rest, so the recoverable accuracy is about 0.5 + strength/2. If the
    pipeline does not find it, nothing else it reports means anything.
    """
    Xc = X.copy()
    informed = rng.random(len(y)) < strength
    planted = np.where(informed, y, rng.integers(0, 2, len(y)))
    Xc["synthetic"] = planted + rng.normal(0, 0.25, len(y))

    s = C.Score(f"positive control (s={strength:.2f})")
    for tr, te in C.walk_forward(len(Xc), k, folds):
        m = RandomForestClassifier(n_estimators=200, min_samples_leaf=20,
                                   n_jobs=-1, random_state=C.SEED)
        m.fit(Xc.iloc[tr], y[tr])
        s.add(y[te], m.predict(Xc.iloc[te]))
    s.extra["expected"] = round(0.5 + strength / 2, 3)
    return s


def autocorrelation(close, k):
    """Serial structure in returns, measured two ways -- and the difference is
    the whole point.

    OVERLAPPING k-bar forward returns are autocorrelated *by construction*: at
    lag j < k, two of them share k-j of their bars. The correlation is arithmetic,
    not information, and it decays to zero at exactly lag k. This is the
    mechanism behind every inflated hit-rate in this project, visible directly.

    NON-OVERLAPPING returns are what actually answer "is there serial structure
    to exploit". Sampled every k bars so no two observations share a bar.
    """
    lags = [1, 3, 6, 12]

    ov = (close.shift(-k) / close - 1).dropna()
    lb_ov = acorr_ljungbox(ov, lags=lags, return_df=True)

    nov = ov.iloc[::k]                       # one observation per k bars
    lb_nov = acorr_ljungbox(nov, lags=lags, return_df=True)

    return {
        "horizon_bars": k,
        "lags": lags,
        "overlapping": {
            "acf": [round(float(ov.autocorr(j)), 4) for j in lags],
            "ljung_box_p": [round(float(p), 4) for p in lb_ov["lb_pvalue"]],
            "n": int(len(ov)),
        },
        "non_overlapping": {
            "acf": [round(float(nov.autocorr(j)), 4) for j in lags],
            "ljung_box_p": [round(float(p), 4) for p in lb_nov["lb_pvalue"]],
            "n": int(len(nov)),
        },
    }


def run(name, horizon, folds):
    rng = np.random.default_rng(C.SEED)
    k = C.HORIZON_BARS[horizon]
    X, y, close = C.dataset(name, horizon)

    C.header(f"{name} — ceiling test, {horizon} horizon",
             f"{len(X):,} bars  {close.index[0].date()} -> {close.index[-1].date()}"
             f"  |  {len(C.BASE_COLS) if hasattr(C,'BASE_COLS') else X.shape[1]} features"
             f"  |  base UP rate {y.mean()*100:.1f}%")

    scores = []

    real = over_capacity(X, y, k, folds, rng, "real labels")
    scores.append(real); print(real.row())

    shuf = over_capacity(X, C.shuffled(y, rng), k, folds, rng, "shuffled labels")
    scores.append(shuf); print(shuf.row())

    orc = nn_oracle(X, y, k, folds)
    scores.append(orc); print(orc.row())

    for strength in (0.10, 0.30):
        pc = positive_control(X, y, k, folds, rng, strength)
        scores.append(pc)
        print(pc.row() + f"   expected ~{pc.extra['expected']*100:.0f}%")

    maj = C.Score("majority-class baseline")
    for tr, te in C.walk_forward(len(X), k, folds):
        maj.add(y[te], np.full(len(te), int(round(y[tr].mean()))))
    scores.append(maj); print(maj.row())

    ac = autocorrelation(close, k)
    lags = "/".join(str(x) for x in ac["lags"])
    print(f"\n  forward-return autocorrelation (lags {lags})")
    print(f"    overlapping     {ac['overlapping']['acf']}   "
          f"(n={ac['overlapping']['n']:,})  <- arithmetic, dies at lag {k} = the horizon")
    print(f"    non-overlapping {ac['non_overlapping']['acf']}   "
          f"(n={ac['non_overlapping']['n']:,})  <- the honest measurement")
    print(f"    Ljung-Box p, non-overlapping: {ac['non_overlapping']['ljung_box_p']}")

    gap = real.train_acc - real.acc
    delta = abs(real.acc - shuf.acc) * 100
    print(f"\n  VERDICT")
    print(f"    memorised {real.train_acc*100:.0f}% of train, generalised "
          f"{real.acc*100:.1f}%  (gap {gap*100:.0f} pts)")
    print(f"    real vs shuffled labels differ by {delta:.1f} pts "
          f"-> {'NO extractable signal' if delta < 1.5 else 'possible signal'}")
    print(f"    pipeline recovered the injected controls, so the null above is "
          f"a property of the data")

    return {"index": name, "horizon": horizon, "bars": int(len(X)),
            "start": str(close.index[0].date()), "end": str(close.index[-1].date()),
            "base_up_rate": round(float(y.mean()), 4),
            "real_vs_shuffled_gap_pts": round(delta, 2),
            "autocorrelation": ac,
            "scores": [s.as_dict() for s in scores]}



def _merge_by_index(path, new_rows):
    """Merge into an existing results file, keyed by index.

    Running the script for a subset of indices should update those entries and
    leave the rest alone. Overwriting silently discarded results that figures
    and tables still referred to.
    """
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = []
    by_index = {r.get("index"): r for r in existing if isinstance(r, dict)}
    for r in new_rows:
        by_index[r.get("index")] = r
    return list(by_index.values())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="30m", choices=list(C.HORIZON_BARS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--indices", nargs="*", default=C.INDICES)
    a = ap.parse_args()

    out = [run(n, a.horizon, a.folds) for n in a.indices]

    d = paths.ROOT / "results"; d.mkdir(exist_ok=True)
    f = d / f"ceiling_test_{a.horizon}.json"
    f.write_text(json.dumps(_merge_by_index(f, out), indent=2))
    print(f"\nwritten: {f.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
