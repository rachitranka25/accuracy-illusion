# 07 — Edges decay

**Claim tested.** That a signal which backtests profitably after costs can be
built into a product.

**Verdict.** Not without checking whether it is still alive. The single best
cost-positive signal this project found was **already dead** by the time it was
discovered — it earned its entire profit in the first year and has been losing
for the last six months of the sample.

---

## The signal

PCR extremes, contrarian, NIFTY only. Trade only when the put-call ratio sits in
its top or bottom quintile; high PCR implies next-day up. Tested per-day, so no
overlapping-sample inflation.

On the full 2-year sample it was the first thing in the entire investigation to
come out positive after costs:

| Threshold | Trades | Win rate | Profit factor | Net (1 lot futures) |
|---|---|---|---|---|
| 80 / 20 | 204 | 53% | **1.11** | **+₹100,000** |
| 90 / 10 | fewer | — | **1.20** | — |

Directional accuracy at the extremes: 58–61%. BANKNIFTY showed nothing robust —
the signal is NIFTY-specific.

> Status: **REPRODUCIBLE** — `research/archive/oi_signals_test.py`.

## The robustness test that killed it

Three checks, any one of which is disqualifying.

### 1. Cost sensitivity

| Round-trip cost | Profit factor |
|---|---|
| < 14 pts | > 1.0 |
| **≥ 16 pts** | **< 1.0** |

Profitable on futures costs. Dead on realistic **options** costs, where theta
plus spread runs 15–30 points. The intended trading vehicle is options.

### 2. Decay over time

| Period | Directional accuracy |
|---|---|
| 2024 | ~74% |
| 2025 | choppy |
| 2026 | **38–43%** |

### 3. The last six months

| | Result |
|---|---|
| Directional accuracy | **36%** |
| Profit factor | **0.74** |

Currently losing. The +₹100,000 was almost entirely earned in 2024.

## Why it decayed

PCR-extreme contrarian trading is well known and widely published. A signal that
is both public and simple gets arbitraged. There is nothing surprising here
except how fast it happened — under two years from clearly profitable to clearly
losing.

## What this changes

**It invalidates the "sustained accuracy" claim as a category.** A claim of the
form "85% over 76 days" carries no information even if every number in it is
true, because a genuinely real edge in this data died inside that horizon. A
76-day window is not long enough to distinguish a live edge from a dying one,
let alone from the measurement artefacts in
[finding 02](02-measurement-illusions.md).

**It changes what a system should be.** A static signal library is the wrong
architecture. What survives is:

- continuous re-testing of signals against recent data, with explicit
  live-or-dead status per signal;
- risk management, which does not decay;
- honest measurement, which does not decay.

**It sets a rule for anything found later.** Every positive result in this
repository must be reported with its performance in the most recent window, not
just the full sample. A full-sample profit factor above 1.0 that is entirely
front-loaded is a dead signal being reported as a live one. This standard is
applied to the variance risk premium in
[finding 05](05-variance-risk-premium.md) and to the pinning lead in
[finding 06](06-open-interest.md).
