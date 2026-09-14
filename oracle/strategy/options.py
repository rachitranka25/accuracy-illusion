#!/usr/bin/env python3
"""
Options Strategy Engine — the Kirk-Du-Plessis game (probability + premium, not
direction). Takes our volatility forecast (BIG move vs QUIET, ~68-74% OOS) and
recommends the right options structure, with real Black-Scholes numbers:
breakevens, max profit / max loss, probability-of-profit, and position size.

  BIG move predicted  -> BUY volatility: Long Straddle / Strangle
      (profit if the move is large, either direction — direction-free)
  QUIET predicted     -> SELL volatility: Iron Condor (defined-risk premium)
      (profit from time decay if price stays in a range)

IV input = India VIX (literally the implied vol of Nifty ATM options). BankNifty
IV is scaled up (~1.15x) since it's structurally more volatile.
"""
import math
from datetime import date

R = 0.065          # India risk-free ~6.5%
LOTS = {"NIFTY50": 75, "BANKNIFTY": 15, "FINNIFTY": 65}
IV_MULT = {"NIFTY50": 1.00, "BANKNIFTY": 1.15, "FINNIFTY": 1.10}


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, T, iv, call=True):
    """Black-Scholes price + delta."""
    if T <= 0 or iv <= 0:
        intr = max(0, (S - K) if call else (K - S))
        return intr, (1.0 if call and S > K else -1.0 if not call and S < K else 0.0)
    d1 = (math.log(S / K) + (R + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if call:
        px = S * _ncdf(d1) - K * math.exp(-R * T) * _ncdf(d2)
        delta = _ncdf(d1)
    else:
        px = K * math.exp(-R * T) * _ncdf(-d2) - S * _ncdf(-d1)
        delta = -_ncdf(-d1)
    return max(px, 0.0), delta


def _round_strike(x, step):
    return round(x / step) * step


def recommend(name, spot, vix_pct, dte, regime, capital=100000, risk_pct=2.0):
    """Return the recommended options strategy with full numbers."""
    if not spot or spot <= 0 or dte <= 0:
        return None
    iv = (vix_pct / 100.0) * IV_MULT.get(name, 1.0)
    T = dte / 365.0
    lot = LOTS.get(name, 50)
    step = 100 if name == "BANKNIFTY" else 50
    sd = spot * iv * math.sqrt(T)          # 1-sigma expected move (points)
    atm = _round_strike(spot, step)

    out = {"name": name, "spot": round(spot, 1), "iv_pct": round(iv * 100, 1),
           "dte": dte, "expected_move_1sd": round(sd), "regime": regime,
           "atm": atm}

    if regime == "BIG":
        # LONG STRADDLE (buy ATM call + put) — profit from a big move either way
        c, _ = bs(spot, atm, T, iv, True)
        p, _ = bs(spot, atm, T, iv, False)
        debit = c + p
        be_up, be_dn = atm + debit, atm - debit
        # POP ≈ P(|move| > debit) under the (higher) predicted vol
        z = debit / sd if sd else 9
        pop = 2 * (1 - _ncdf(z)) * 100
        out.update({
            "play": "LONG STRADDLE (buy volatility)",
            "why": "A big move is likely — you win whichever way it breaks.",
            "legs": [f"BUY {atm} CE @ ~{c:.0f}", f"BUY {atm} PE @ ~{p:.0f}"],
            "cost_per_lot": round(debit * lot), "premium_pts": round(debit),
            "breakevens": [round(be_dn), round(be_up)],
            "max_loss": round(debit * lot), "max_profit": "unlimited (big move)",
            "pop_pct": round(pop),
            "note": "Book when the move happens — don't let it fade. Loss capped at the premium paid.",
        })
    else:  # QUIET
        # IRON CONDOR: sell ~1SD strangle, buy ~2SD wings (defined risk)
        sc_k = _round_strike(spot + sd, step); sp_k = _round_strike(spot - sd, step)
        lc_k = _round_strike(spot + 2 * sd, step); lp_k = _round_strike(spot - 2 * sd, step)
        sc, _ = bs(spot, sc_k, T, iv, True); sp, _ = bs(spot, sp_k, T, iv, False)
        lc, _ = bs(spot, lc_k, T, iv, True); lp, _ = bs(spot, lp_k, T, iv, False)
        credit = (sc + sp) - (lc + lp)
        width = (lc_k - sc_k)
        max_loss = max(width - credit, 0)
        be_up, be_dn = sc_k + credit, sp_k - credit
        # POP ≈ P(price stays between short strikes)
        zu = (sc_k - spot) / sd if sd else 9
        zd = (sp_k - spot) / sd if sd else -9
        pop = (_ncdf(zu) - _ncdf(zd)) * 100
        out.update({
            "play": "IRON CONDOR (sell volatility · defined risk)",
            "why": "A quiet range is likely — collect premium as time decays.",
            "legs": [f"SELL {sc_k} CE @ ~{sc:.0f}", f"SELL {sp_k} PE @ ~{sp:.0f}",
                     f"BUY {lc_k} CE @ ~{lc:.0f}", f"BUY {lp_k} PE @ ~{lp:.0f}"],
            "credit_per_lot": round(credit * lot), "credit_pts": round(credit),
            "breakevens": [round(be_dn), round(be_up)],
            "max_loss": round(max_loss * lot), "max_profit": round(credit * lot),
            "pop_pct": round(pop),
            "note": "Wings cap the risk (defined). Exit at ~50% of max profit or if price nears a short strike.",
        })

    # position sizing on the user's risk budget
    risk_budget = capital * risk_pct / 100
    ml = out["max_loss"] if isinstance(out["max_loss"], (int, float)) else risk_budget
    out["suggested_lots"] = max(1, int(risk_budget / ml)) if ml > 0 else 1
    out["risk_budget"] = round(risk_budget)
    return out


if __name__ == "__main__":
    import json
    # demo: Nifty 24350, VIX 13%, weekly expiry (3 DTE)
    print(json.dumps(recommend("NIFTY50", 24350, 13, 3, "BIG"), indent=2))
    print(json.dumps(recommend("BANKNIFTY", 57000, 13, 3, "QUIET"), indent=2))
