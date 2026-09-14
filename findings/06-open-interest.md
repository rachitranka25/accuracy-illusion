# 06 — Does the option chain know something price does not?

**Claim tested.** That open interest and participant positioning carry
information about the next day's move that is absent from the price series.

**Verdict.** Yes, but barely. Over 2,181 trading days the option-chain
residual carries a small, statistically stable signal — information coefficient
0.04–0.06, hit-rate 52% — while the price-only baseline carries nothing at all
on the same days. Real, and too small to trade alone.

---

## Why this was worth asking

[Finding 01](01-directional-ceiling.md) established that the model's thirteen
features are all transforms of one price series. Open interest is different in
kind: it is what market participants have actually committed money to, and it is
free in the EOD bhavcopy.

Every feature below is built from data available at the **close of day T** and
used only to predict **day T+1**. The one-day shift is the only thing standing
between this test and a lie, so it happens in exactly one place in the code.

## First attempt, and the trap it revealed

On 350 days, `dist_maxpain` — the distance from spot to the max-pain strike —
predicted next-day open-to-close at about 60%.

It was then correlated against plain price features. **`dist_maxpain` is 0.818
correlated with the 10-day return**, and the 10-day return alone did the same job
slightly better.

The apparent option-chain edge was price mean-reversion wearing an options
costume. Three features (`dist_maxpain`, `dist_res`, `dist_sup`) were also three
views of one idea and moved together, so counting them as three discoveries
would have tripled the apparent evidence.

**Method correction adopted:** correlated features get combined into a single
z-score, and any features-versus-price claim must be tested on the **residual**
after the price component is regressed out.

## The decade test

350 days can reject a feature but cannot distinguish "no signal" from "signal
too small to see". A decade of EOD option chains was collected
(`oracle/data/option_chain.py` → `data_cache/{name}_oc_long.csv`) and the
sharper question asked: **strip out everything the price series already explains
— does the OI residual predict anything on its own?**

BANKNIFTY, 2,181 days, 27 May 2016 → 26 May 2026. Target: next-day open-to-close.

### Raw features

| Feature | IC | p | hit% | H1 | H2 | Verdict |
|---|---|---|---|---|---|---|
| pcr_oi | 0.038 | 0.080 | 51.8 | 0.070 | 0.015 | no |
| pcr_doi | 0.045 | 0.036 | 53.0 | 0.094 | −0.008 | unstable |
| dist_maxpain | −0.017 | 0.439 | 51.9 | −0.001 | −0.034 | no |
| dist_sup | −0.045 | 0.037 | 52.6 | −0.047 | −0.041 | **keep** |
| d_pcr_oi | 0.051 | 0.017 | 51.5 | 0.076 | 0.022 | unstable |

### Residual (price component removed)

| Feature | IC | p | hit% | H1 | H2 | Verdict |
|---|---|---|---|---|---|---|
| **d_pcr_oi** | **0.059** | **0.006** | 51.3 | 0.052 | 0.068 | **keep** |
| pcr_oi | 0.044 | 0.041 | 51.9 | 0.062 | 0.045 | **keep** |
| pcr_vol | 0.043 | 0.043 | 51.9 | 0.055 | 0.040 | **keep** |
| dist_maxpain | −0.045 | 0.036 | 52.5 | −0.034 | −0.050 | **keep** |

### Price-only baseline, same days

| Feature | IC | p | hit% | Verdict |
|---|---|---|---|---|
| ret1 … ret20 | −0.014 to 0.001 | 0.51–0.97 | 48.2–53.0 | no |
| vol20 | −0.030 | 0.160 | 50.6 | no |

**Nothing in the price-only set survives.** Four residual option-chain features
do, with consistent sign across both halves of a decade.

> Status: **REPRODUCIBLE** — `research/test_oc_residual.py`.

## How to read an IC of 0.05

An information coefficient of 0.04–0.06 is not embarrassing — it is in the range
of genuine daily equity factors, which is why it is detectable over 2,181 days
and invisible over 350. It corresponds to roughly a 52% hit-rate.

Whether 52% is tradeable is answered by [finding 03](03-cost-geometry.md):
at a symmetric payoff, no. It is a **context input**, not a standalone signal.

Notably, `d_pcr_oi` — the *change* in put-call ratio, i.e. today's positioning
flow rather than its level — is the strongest and the only one that strengthens
in the second half. Flow beats level. That is consistent with everything else in
this project: what people did today carries more than what they are holding.

## Pinning: the one result that looks tradeable

If distance from the OI cluster mean-reverts, the rule writes itself: fade the
stretch. Combine `dist_maxpain`, `dist_res` and `dist_sup` into one z-score;
enter only when the stretch exceeds 1σ; short when spot is far above the
cluster, long when far below. Nothing fitted — the rule is fixed in advance and
the training window only sets the z-score scale.

BANKNIFTY, 350 days, first 60% in-sample, last 40% untouched:

| | n | hit% | avg | total | Sharpe |
|---|---|---|---|---|---|
| In-sample | 23 | 65.2% | +0.221% | +5.08% | 5.66 |
| **Out-of-sample** | **55** | **60.0%** | **+0.211%** | **+11.59%** | **3.23** |
| Baseline (always long) | 139 | 48.2% | +0.058% | — | — |

> Status: **REPRODUCIBLE** — `research/test_pinning.py`.

### Why this is not being called an edge

Held against the standard this project applies to everyone else's numbers:

1. **n = 55.** At a true 52% rate, 60% over 55 trades is well inside normal
   variance. This is the same sample-size problem that made the NIFTY 50
   volatility-filtered ORB result unusable in [finding 04](04-volatility-predictability.md).
2. **The decade test disagrees.** The same `dist_maxpain` family scores 52.5%
   over 2,181 days. The honest reading is that the true rate is near 52% and the
   60% is the lucky end of a short window.
3. **The out-of-sample period is one regime** — 29 December 2025 to 24 July
   2026, a single seven-month stretch.
4. **Costs are not in it.** +0.211% per trade on BANKNIFTY is roughly 120 points,
   which *would* clear the ~27-point round trip — this is the one result in the
   project where the cost geometry is not automatically fatal. That is what makes
   it worth continuing rather than closing.

**Status: the most promising open lead in the project**, and the correct next
step is to run the identical fixed rule on the decade of chain data now
available, where 55 trades becomes several hundred. That test has not been run.

## Participant positioning (FII / DII / Pro / Client)

From NSE participant-wise OI, 2 years (`oracle/data/participant_oi.py`).

The rule: fade the FII when its net index-futures position deviates from its
20-day average — more bullish than usual implies next-day down.

| Index | Next-day accuracy | Stability | After cost |
|---|---|---|---|
| BANKNIFTY | **~60%** | 61 / 59 | +₹29k over 2 years, 1 lot |
| NIFTY 50 | ~55% | weaker | — |

Positional, not intraday. Wired into the dashboard as a daily-bias line only.

> Status: **RECORDED** (1 Aug 2026) — `oracle/strategy/smart_money.py`.
> An unverified claim exists that a momentum version (FII buying 3+ consecutive
> days → 65% up over the next week) works better. It has not been tested and is
> a different horizon. Do not cite it.

## Intraday OI — collected, not yet analysed

EOD open interest turned out to be a slow proxy for price (the 0.818
correlation). Intraday OI is a different object: it says whether positions are
being built or unwound *while the session runs*, which the price series does not
carry.

`oracle/data/intraday_oi.py` collects it at 10-minute resolution. The source
keeps only a rolling month, so the file appends and the local history grows past
what the site itself will serve. **This is an accumulating asset with no result
yet** — it is the cleanest untested hypothesis left in the project.
