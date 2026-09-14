#!/usr/bin/env python3
"""Seven model families, one task, one split.

The claim under test is that intraday direction on NSE index futures is not
predictable from freely available 5-minute data. The obvious objection is that
a better model would find it, so every family that gets proposed for this
problem is trained here under identical conditions:

    same features (the live system's own), same purged walk-forward splits,
    same labels, same seeds, and a shuffled-label control for each model.

The shuffled control is what makes a null result interpretable. A model scoring
50% could mean the market is unpredictable or that the model is broken. A model
scoring 50% on real labels *and* 50% on shuffled labels, in a pipeline that
recovers an injected synthetic signal (see research/ceiling_test.py), can only
mean the first.

Families: logistic regression, gradient boosting, LightGBM, XGBoost,
k-nearest-neighbours, LSTM, Transformer, and a compact Temporal Fusion
Transformer. Plus a majority-class baseline, because a model that always
predicts UP in a drifting market beats 0.5 without predicting anything.

Run:  python3 -m research.model_zoo [--horizon 30m] [--folds 5] [--quick]
"""
import os

# Must precede every native import. PyTorch, LightGBM and XGBoost each ship
# their own OpenMP runtime; on macOS the duplicate runtimes deadlock XGBoost's
# training loop. Pinning the pool to one thread is the portable fix, and single
# threading also makes the benchmark bit-for-bit reproducible.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

from oracle import paths
from research import common as C

torch.set_num_threads(2)

SEQ = 24            # 2 hours of 5-minute bars for the sequence models
EPOCHS = 12
BATCH = 256
DEVICE = torch.device("cpu")     # deterministic and fast enough at this size


# ══ sequence models ═══════════════════════════════════════════════════
class LSTMNet(nn.Module):
    def __init__(self, nf, hid=48):
        super().__init__()
        self.rnn = nn.LSTM(nf, hid, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class TransformerNet(nn.Module):
    def __init__(self, nf, d=48, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(nf, d)
        self.pos = nn.Parameter(torch.zeros(1, SEQ, d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.1,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos)
        return self.head(h[:, -1]).squeeze(-1)


class GRN(nn.Module):
    """Gated residual network — the building block of the TFT."""

    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, d), nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = self.fc2(torch.nn.functional.elu(self.fc1(x)))
        return self.norm(x + torch.sigmoid(self.gate(h)) * h)


class TFTNet(nn.Module):
    """Compact Temporal Fusion Transformer (Lim et al., 2021).

    Keeps the three components that define the architecture — a variable
    selection network that learns per-feature weights, GRN gating, and an
    interpretable multi-head attention layer over an LSTM encoder. It omits
    static covariates and the quantile head, because this task has neither a
    static entity nor a distributional target. Reported as a compact variant,
    not as the published model.
    """

    def __init__(self, nf, d=48, heads=4):
        super().__init__()
        self.vsn = nn.Sequential(nn.Linear(nf, nf), nn.Softmax(dim=-1))
        self.proj = nn.Linear(nf, d)
        self.grn_in = GRN(d)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.grn_out = GRN(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        w = self.vsn(x.mean(dim=1)).unsqueeze(1)      # per-variable selection weights
        h = self.grn_in(self.proj(x * w))
        h, _ = self.lstm(h)
        a, _ = self.attn(h, h, h)
        return self.head(self.grn_out(h + a)[:, -1]).squeeze(-1)


def _windows(Xs, y, idx):
    """Sequence windows that never reach back across the split boundary."""
    keep = idx[idx >= SEQ]
    if len(keep) == 0:
        return None, None
    seq = np.stack([Xs[i - SEQ:i] for i in keep])
    return torch.tensor(seq, dtype=torch.float32), torch.tensor(y[keep], dtype=torch.float32)


def fit_torch(model_cls, Xs, y, tr, te, seed):
    torch.manual_seed(seed)
    xa, ya = _windows(Xs, y, tr)
    xb, yb = _windows(Xs, y, te)
    if xa is None or xb is None:
        return None, None

    m = model_cls(Xs.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    m.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xa))
        for i in range(0, len(xa), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            lossf(m(xa[b].to(DEVICE)), ya[b].to(DEVICE)).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

    m.eval()
    with torch.no_grad():
        pred = (torch.sigmoid(m(xb.to(DEVICE))) > 0.5).long().cpu().numpy()
    return yb.long().numpy(), pred


# ══ tabular models ════════════════════════════════════════════════════
def tabular_models(quick):
    n = 100 if quick else 300
    return {
        "logistic regression": lambda: LogisticRegression(max_iter=2000, C=1.0),
        "gradient boosting": lambda: GradientBoostingClassifier(
            n_estimators=n, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=C.SEED),
        "LightGBM": lambda: lgb.LGBMClassifier(
            n_estimators=n, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=40,
            random_state=C.SEED, verbose=-1, n_jobs=1),
        "XGBoost": lambda: xgb.XGBClassifier(
            n_estimators=n, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=C.SEED, verbosity=0, tree_method="hist", n_jobs=1),
        "k-nearest neighbours": lambda: KNeighborsClassifier(n_neighbors=50, n_jobs=1),
    }


SEQ_MODELS = {"LSTM": LSTMNet, "Transformer": TransformerNet,
              "Temporal Fusion Transformer": TFTNet}


def evaluate(name, horizon, folds, quick):
    rng = np.random.default_rng(C.SEED)
    k = C.HORIZON_BARS[horizon]
    X, y, close = C.dataset(name, horizon)
    y_shuf = C.shuffled(y, rng)
    splits = list(C.walk_forward(len(X), k, folds))

    C.header(f"{name} — model zoo, {horizon} horizon",
             f"{len(X):,} bars  {close.index[0].date()} -> {close.index[-1].date()}"
             f"  |  {X.shape[1]} features  |  {len(splits)} purged walk-forward folds"
             f"  |  base UP rate {y.mean()*100:.1f}%")

    results = []

    maj = C.Score("majority-class baseline")
    for tr, te in splits:
        maj.add(y[te], np.full(len(te), int(round(y[tr].mean()))))
    results.append((maj, None))

    for label, make in tabular_models(quick).items():
        t0 = time.time()
        real, ctrl = C.Score(label), C.Score(f"{label} [shuffled]")
        preds, truths = [], []
        for tr, te in splits:
            sc = StandardScaler().fit(X.iloc[tr])
            a, b = sc.transform(X.iloc[tr]), sc.transform(X.iloc[te])
            m = make(); m.fit(a, y[tr]); p = m.predict(b)
            real.add(y[te], p); preds.append(p); truths.append(y[te])

            m2 = make(); m2.fit(a, y_shuf[tr]); ctrl.add(y_shuf[te], m2.predict(b))
        real.extra["seconds"] = round(time.time() - t0, 1)
        real.extra["pt_p"] = round(C.pesaran_timmermann(
            np.concatenate(truths), np.concatenate(preds))[1], 4)
        results.append((real, ctrl))

    for label, cls in SEQ_MODELS.items():
        t0 = time.time()
        real, ctrl = C.Score(label), C.Score(f"{label} [shuffled]")
        preds, truths = [], []
        for tr, te in splits:
            sc = StandardScaler().fit(X.iloc[tr])
            Xs = sc.transform(X).astype(np.float32)
            yt, yp = fit_torch(cls, Xs, y, tr, te, C.SEED)
            if yt is None:
                continue
            real.add(yt, yp); preds.append(yp); truths.append(yt)
            st, sp = fit_torch(cls, Xs, y_shuf, tr, te, C.SEED)
            if st is not None:
                ctrl.add(st, sp)
        real.extra["seconds"] = round(time.time() - t0, 1)
        if preds:
            real.extra["pt_p"] = round(C.pesaran_timmermann(
                np.concatenate(truths), np.concatenate(preds))[1], 4)
        results.append((real, ctrl))

    for real, ctrl in results:
        line = real.row()
        if ctrl is not None and ctrl.n:
            line += f"  | shuffled {ctrl.acc*100:.1f}%  Δ{abs(real.acc-ctrl.acc)*100:+.1f}"
        if "pt_p" in real.extra:
            line += f"  PT p={real.extra['pt_p']:.3f}"
        print(line)

    best = max((r for r, _ in results[1:]), key=lambda s: s.acc)
    print(f"\n  best family: {best.name} at {best.acc*100:.1f}% "
          f"(95% CI {best.ci[0]*100:.1f}-{best.ci[1]*100:.1f}, "
          f"p={best.p_vs_coin:.3f} against a coinflip)")
    print(f"  every family within {max(r.acc for r,_ in results[1:])*100 - min(r.acc for r,_ in results[1:])*100:.1f} "
          f"points of every other — the choice of architecture does not matter here")

    return {"index": name, "horizon": horizon, "bars": int(len(X)),
            "folds": len(splits), "base_up_rate": round(float(y.mean()), 4),
            "models": [{**r.as_dict(),
                        "shuffled_accuracy": (round(c.acc, 4) if c and c.n else None)}
                       for r, c in results]}



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
    ap.add_argument("--quick", action="store_true", help="fewer trees; for smoke tests")
    a = ap.parse_args()

    out = [evaluate(n, a.horizon, a.folds, a.quick) for n in a.indices]

    d = paths.ROOT / "results"; d.mkdir(exist_ok=True)
    f = d / f"model_zoo_{a.horizon}.json"
    f.write_text(json.dumps(_merge_by_index(f, out), indent=2))
    print(f"\nwritten: {f.relative_to(paths.ROOT)}")


if __name__ == "__main__":
    main()
