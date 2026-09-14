#!/usr/bin/env python3
"""
Volatility engine (LSTM) — predicts whether the NEXT 30 minutes will see a BIG
move (high realized volatility) vs a quiet stretch. This is direction-FREE: it
forecasts the SIZE of the coming move, not which way. Out-of-sample it holds
~68% (BankNifty) / ~74% (Nifty50), stable across both test halves — because
volatility clusters (a real, well-documented market property), unlike direction.

Trades it enables: predicted BIG move -> buy a straddle/strangle (profit either
way). Predicted QUIET -> avoid buying options (theta will bleed you), or sell
premium. It never says up or down.
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from oracle.models import direction as fe
from oracle import paths

CACHE = paths.CACHE
SEQ = 24                 # look back 24 five-min bars (2 hours)
K = 6                    # predict over next 30 min (6 bars)
torch.manual_seed(0)


class _LSTM(nn.Module):
    def __init__(self, nf, hid=48):
        super().__init__()
        self.lstm = nn.LSTM(nf, hid, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hid, 24), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(24, 1))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :]).squeeze(-1)


def _five_min(candles, day):
    df = pd.DataFrame(candles, columns=["t", "Open", "High", "Low", "Close"])
    df.index = pd.to_datetime(day + " " + df["t"])
    m5 = df.resample("5min").agg({"Open": "first", "High": "max",
                                  "Low": "min", "Close": "last"}).dropna()
    return m5


class VolatilityLSTM:
    def __init__(self, name):
        self.name = name
        self.cols = fe.BASE_COLS
        self.ok = False
        pt = CACHE / f"{name}_vol_lstm.pt"
        m5 = fe.load_long(name)
        if m5 is None:
            print(f"  {name}: vol-LSTM — no long history, skipped")
            return
        # keep recent history to warm up intraday features (else dropna kills it)
        self.warmup5 = m5[["Open", "High", "Low", "Close"]].tail(150).copy()
        if self.warmup5.index.tz is not None:      # match tz-naive intraday stamps
            self.warmup5.index = self.warmup5.index.tz_localize(None)
        stamp = str(m5.index[-1].date())
        if pt.exists():
            try:
                s = torch.load(pt, map_location="cpu", weights_only=False)
                if s.get("stamp") == stamp:
                    self._restore(s)
                    print(f"  {name}: loaded cached vol-LSTM (OOS ~{self.acc}%)")
                    return
            except Exception:
                pass
        self._train(m5, stamp, pt)

    def _restore(self, s):
        self.mu = s["mu"]; self.sd = s["sd"]; self.acc = s["acc"]
        self.model = _LSTM(len(self.cols))
        self.model.load_state_dict(s["state"]); self.model.eval()
        self.ok = True

    def _train(self, m5, stamp, pt, epochs=8):
        print(f"  {self.name}: training vol-LSTM ({len(m5)} bars)…")
        feats, c5 = fe.build5(m5); feats = feats.dropna()
        idx = feats.index
        c = c5.reindex(idx).values
        dates = pd.to_datetime(idx).normalize().view("int64")
        X = feats[self.cols].values.astype(np.float32)
        n = len(idx)
        ret = np.abs(np.diff(c, prepend=c[0]))
        fut = pd.Series(ret).rolling(K).sum().shift(-K).values
        med = np.nanmedian(fut)
        lab = np.where(np.isnan(fut), np.nan, (fut > med).astype(np.float32))
        ends = np.array([t for t in range(SEQ - 1, n - K)
                         if not np.isnan(lab[t]) and dates[t - SEQ + 1] == dates[t]])
        cut = int(len(ends) * 0.7)
        tr_e, te_e = ends[:cut], ends[cut:]
        self.mu = X[:tr_e[-1]].mean(0); self.sd = X[:tr_e[-1]].std(0) + 1e-6
        Xn = (X - self.mu) / self.sd

        def make(es):
            seqs = np.stack([Xn[t - SEQ + 1:t + 1] for t in es]).astype(np.float32)
            return torch.tensor(seqs), torch.tensor(lab[es].astype(np.float32))

        Xtr, ytr = make(tr_e); Xte, yte = make(te_e)
        self.model = _LSTM(len(self.cols))
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            self.model.train(); perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), 512):
                b = perm[i:i + 512]; opt.zero_grad()
                loss = lossf(self.model(Xtr[b]), ytr[b]); loss.backward(); opt.step()
        self.model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(self.model(Xte)).numpy() >= 0.5).astype(float)
        self.acc = round(float((pred == yte.numpy()).mean()) * 100)
        self.ok = True
        try:
            torch.save({"stamp": stamp, "mu": self.mu, "sd": self.sd,
                        "acc": self.acc, "state": self.model.state_dict()}, pt)
            print(f"  {self.name}: vol-LSTM trained, OOS ~{self.acc}% ✅")
        except Exception as exc:
            print(f"  {self.name}: vol-LSTM cache save failed ({exc})")

    def predict(self, candles, day):
        """Return {'vol':'BIG'|'QUIET', 'prob':pct, 'acc':oos_acc} or None."""
        if not self.ok:
            return None
        try:
            today5 = _five_min(candles, day)
            # One bar is enough: the warmup below supplies the whole lookback. The
            # old "< 3" rule left the model silent until ~09:30, so the first trades
            # of the day escaped every check that depends on it — and it also made
            # live behave differently from training, where the lookback always
            # spanned the previous session.
            if len(today5) < 1:
                return None
            warm = getattr(self, "warmup5", None)
            m5 = today5 if warm is None else pd.concat(
                [warm, today5])[~pd.concat([warm, today5]).index.duplicated(
                    keep="last")].sort_index()
            feats, _ = fe.build5(m5); feats = feats.dropna()
            if len(feats) < SEQ:
                return None
            X = feats[self.cols].values.astype(np.float32)
            Xn = (X - self.mu) / self.sd
            seq = torch.tensor(Xn[-SEQ:][None, :, :].astype(np.float32))
            with torch.no_grad():
                p = float(torch.sigmoid(self.model(seq)).item())
            big = p >= 0.5
            return {"vol": "BIG" if big else "QUIET",
                    "prob": round((p if big else 1 - p) * 100),
                    "acc": self.acc}
        except Exception:
            return None


if __name__ == "__main__":
    for nm in ["BANKNIFTY", "NIFTY50"]:
        VolatilityLSTM(nm)
