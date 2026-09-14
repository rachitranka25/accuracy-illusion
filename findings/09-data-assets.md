# 09 — What a retail account can actually get for free

**Claim tested.** That serious research on NSE index derivatives requires paid
data.

**Verdict.** Less true than assumed. A decade of EOD option chains, two years of
per-strike open interest, two years of participant-wise positioning and real
futures volume are all free and all obtainable from a retail setup. What remains
genuinely paid is intraday per-strike OI and historical order flow.

This finding exists because the project spent weeks assuming a wall that was not
there.

---

## What is on disk

| Asset | Coverage | Rows | Source |
|---|---|---|---|
| Index 5-min bars (BANKNIFTY, NIFTY 50) | Feb 2025 → Jul 2026 (17 mo) | 26,265 each | Angel One SmartAPI |
| Index 5-min bars (FINNIFTY, MIDCPNIFTY, SENSEX) | 2025 → 2026 | 9.6k–14k | Angel One SmartAPI |
| **Option chain EOD** (NIFTY) | Jul 2024 → Jul 2026 | **846,540** | NSE F&O bhavcopy |
| **Option chain EOD** (BANKNIFTY) | Jul 2024 → Jul 2026 | **536,198** | NSE F&O bhavcopy |
| **Long-history option chain** (BANKNIFTY) | **May 2016 → May 2026 (10 yr)** | 243,556 | Public archive |
| Participant-wise OI (FII / DII / Pro / Client) | Jul 2024 → Jul 2026 | 515 days | NSE participant file |
| Futures OHLC **with real volume** | Jan 2024 → Jul 2026 daily; ~3 mo of 5-min | 639 daily | Angel One (NFO) |
| India VIX 5-min | 2025 → 2026 | — | Angel One |
| Intraday OI (10-min) | Aug 2026 → (accumulating) | 987 | Rolling public feed |
| Order-flow imbalance (1-min) | Aug 2026 → (accumulating) | 958 per index | Angel Level-2 depth on futures |

## The four things that were not obvious

### 1. The bhavcopy archive is not geo-blocked

NSE's **live** option-chain API returns 403/404 from outside India. Its **static
archive** does not. The UDiFF F&O bhavcopy files download normally with an
ordinary browser user-agent, and they contain per-strike OHLC, volume, open
interest and change in open interest.

That single fact turned the project's biggest assumed constraint into a
two-year dataset — and, via a second public archive, a ten-year one.
Implied volatility is not included and must be computed from the premium.

`oracle/data/bhavcopy.py`, `oracle/data/option_chain.py`

### 2. Index spot has no volume; futures do

Index spot prints volume = 0. Every VWAP-derived feature computed on spot is
therefore an unweighted expanding mean of typical price — not a volume-weighted
average, despite the name. This was live in the feature set for weeks before it
was caught, and it was caught by an external reviewer rather than internally.

The futures contract carries real volume, which is the fix. Angel provides ~2.5
years of daily futures but only ~3 months of 5-minute data, because a contract
does not live longer than that and expired contracts are not served.

`oracle/data/futures.py`

### 3. TOTP authentication works from anywhere; SMS does not

Angel One's SmartAPI authenticates by time-based one-time password, computed
locally. It works from any country with no phone dependency. Upstox requires an
SMS OTP to an Indian number for daily login, which made it unusable for this
project and the integration was deleted.

For anyone building from outside India, this is the deciding factor between
brokers, and it is not documented anywhere prominent.

`oracle/data/angel.py`

### 4. Order flow cannot be bought back, only collected forward

Historical Level-2 / order-flow data is paid-only; NSE and SEBI do not
distribute it. But Angel's live API does serve best-5 depth on **futures**
(index spot has no order book), so it can be recorded going forward.

`oracle/data/order_flow.py` computes, per minute:

- **OFI-touch** — level-1 imbalance
- **OFI-5** — five-level imbalance
- **OFI-weighted** — depth-weighted, nearest levels heaviest
- **microprice deviation** — size-weighted fair value versus mid
- **spread**

This matters because the microstructure literature identifies order-flow
imbalance as the one place a genuine short-horizon edge is documented to exist
(Kolm, Turiel & Westray, *Deep Order Flow Imbalance*, 2023). It is also the one
hypothesis in this project that has **not** been tested, purely for want of
history. 958 minutes is not a sample.

## Two accumulating assets, no results yet

Both of these grow only with wall-clock time and both are the honest answer to
"what would you need to actually find something new":

| Asset | Why it is different | Status |
|---|---|---|
| Intraday OI (10-min) | Says whether positions are being *built or unwound* during the session — information the price series does not carry. EOD OI is a slow price proxy (0.818 correlated with the 10-day return); intraday is not. | ~1,000 rows |
| Order-flow imbalance (1-min) | The only short-horizon edge documented in peer-reviewed microstructure work. | ~960 rows per index |

The honest statement about both: **they are the reason to keep the collectors
running, and there is currently nothing to report from either.**

## Reproducing the data layer

```bash
python3 -m oracle.data.index_history SENSEX   # any Angel-listed index
python3 -m oracle.data.futures                # futures OHLC + real volume
python3 -m oracle.data.bhavcopy               # 2yr option + futures EOD with OI
python3 -m oracle.data.option_chain           # decade of EOD option chains
python3 -m oracle.data.participant_oi         # FII / DII / Pro / Client
python3 -m oracle.data.intraday_oi            # run daily; appends
python3 -m oracle.data.order_flow             # run during market hours
```

Everything appends. Nothing expires. The two-year and ten-year windows roll
forward on their own.
