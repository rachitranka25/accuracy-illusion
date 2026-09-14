#!/usr/bin/env python3
"""
AI Technical Analyst — an experienced-trader "mind" reads the live chart aloud.

Backend: NVIDIA-hosted LLM (Nemotron) first, Gemini as fallback. Keys come
env-first (NVIDIA_API_KEY / GEMINI_API_KEY) via envconfig.

Input: pure technicals (price, opening range, prior-day levels, indicators,
patterns, forecast-stream stats, today's trades). No news.

Output: a senior-trader style read — market structure, the levels that matter
right now, what would CONFIRM and what would INVALIDATE each scenario, and one
discipline note. It NEVER says buy/sell and NEVER predicts prices. The edge, if
any, is in the backtested engine; the LLM's job is to make the chart readable —
context, not signals.
"""

import json
import time
from pathlib import Path

import requests

# ---- backends ------------------------------------------------------------
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
GEMINI_MODEL = "gemini-flash-latest"
CACHE_SEC = 600                   # 10-min cache (protects free quotas)


def _nvidia_key():
    from oracle.config import from_env_or_json
    cfg = from_env_or_json("nvidia_config.json", {"api_key": "NVIDIA_API_KEY"})
    return cfg["api_key"].strip() if cfg else None


def _gemini_key():
    from oracle.config import from_env_or_json
    cfg = from_env_or_json("gemini_config.json", {"api_key": "GEMINI_API_KEY"})
    return cfg["api_key"].strip() if cfg else None


def _llm(prompt, temperature=0.3, max_tokens=2000):
    """Unified call: NVIDIA Nemotron first, Gemini fallback. Returns (text, error)."""
    # --- NVIDIA (OpenAI-compatible) ---
    nkey = _nvidia_key()
    if nkey:
        try:
            r = requests.post(
                NVIDIA_URL,
                headers={"Authorization": f"Bearer {nkey}",
                         "Accept": "application/json"},
                json={"model": NVIDIA_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": temperature, "top_p": 1,
                      "max_tokens": max_tokens, "stream": False},
                timeout=60)
            j = r.json()
            if "choices" in j:
                return j["choices"][0]["message"]["content"].strip(), None
            nerr = f"NVIDIA: {str(j)[:120]}"
        except Exception as exc:
            nerr = f"NVIDIA call failed: {str(exc)[:120]}"
    else:
        nerr = "no NVIDIA key"

    # --- Gemini fallback ---
    gkey = _gemini_key()
    if gkey:
        try:
            r = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{GEMINI_MODEL}:generateContent?key={gkey}",
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": temperature,
                                           "maxOutputTokens": max_tokens}},
                timeout=30)
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip(), None
        except Exception as exc:
            return None, f"{nerr}; Gemini also failed: {str(exc)[:100]}"
    return None, nerr


# strip <think>…</think> reasoning some Nemotron variants emit
def _clean(text):
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()


_cache = {"at": 0.0, "text": None}


def briefing(market_state: str, force=False):
    """Senior-trader technical read. Returns (text, error). Cached."""
    if not force and _cache["text"] and time.time() - _cache["at"] < CACHE_SEC:
        return _cache["text"], None
    prompt = f"""You are a trader with 30+ years reading Indian index charts
(BankNifty & Nifty50), mentoring a beginner. Below is the LIVE technical state
from their system (prices, opening range, prior-day levels, indicators, candle
patterns, and their model's stats).

Read the chart the way a seasoned discretionary trader would. Write a SHORT read
(max 200 words) in simple English, using ONLY the data given — never invent numbers:

1. STRUCTURE — one line per index: trending or ranging, and which side of the
   key levels price sits (vs opening range, prior day, VWAP).
2. LEVELS THAT MATTER NOW — the 2-3 most important prices per index, and why.
3. SCENARIOS — for each index: what price action would CONFIRM strength /
   weakness, and what would INVALIDATE it. Describe conditions, not calls.
4. DISCIPLINE NOTE — one line based on the indicators/stats (e.g. conflicting
   signals -> smaller size; trend day -> don't fade it).

Hard rules: NO buy/sell recommendations. NO price targets or predictions. If the
data is mixed or thin, say so plainly. Do not show your reasoning steps — give
only the final read.

LIVE TECHNICAL STATE:
{market_state}"""
    text, err = _llm(prompt, temperature=0.3)
    if text:
        text = _clean(text)
        _cache.update(at=time.time(), text=text)
        return text, None
    return None, err


_copilot_cache = {"at": 0.0, "verdict": None}


def copilot(market_state: str, force=False):
    """Structured synthesis of ALL system outputs. Returns (dict, error)."""
    if not force and _copilot_cache["verdict"] and \
            time.time() - _copilot_cache["at"] < CACHE_SEC:
        return _copilot_cache["verdict"], None
    prompt = f"""You are the co-pilot of an intraday trading system for BankNifty
& Nifty50, thinking like a veteran trader. Below is EVERYTHING the system knows
right now: multi-horizon ML forecasts with live hit-rates, indicators, levels,
opening range, the backtested engine's state, and today's paper trades.

SYNTHESIZE all of it into one structured verdict per index. Weigh the evidence:
do the horizons agree or conflict? Do indicators confirm the model bias? Are
today's live hit-rates good or poor (below ~55% means the models are struggling
today — say so)? Is price at a meaningful level or in no-man's-land?

Respond with ONLY this JSON (no markdown, no reasoning, no extra text):
{{"BANKNIFTY": {{"stance": "LEANS-LONG|LEANS-SHORT|STAND-ASIDE",
  "conviction": 1-5,
  "read": "<=25 words: the strongest evidence, honestly weighed",
  "wait_for": "<=20 words: what concrete price action would strengthen the case"}},
 "NIFTY50": {{...same fields...}}}}

Rules: conviction 4-5 ONLY when horizons, indicators AND hit-rates all agree. If
evidence conflicts, stance=STAND-ASIDE with low conviction — that is a valuable
answer, not a failure. Use only the data given.

SYSTEM STATE:
{market_state}"""
    text, err = _llm(prompt, temperature=0.2)
    if not text:
        return None, err
    try:
        text = _clean(text)
        text = text[text.find("{"):text.rfind("}") + 1]   # isolate JSON
        verdict = json.loads(text)
        _copilot_cache.update(at=time.time(), verdict=verdict)
        return verdict, None
    except Exception as exc:
        return None, f"copilot parse failed: {str(exc)[:100]}"


if __name__ == "__main__":
    text, err = briefing("(test run — no live readings)")
    print(text or f"ERROR: {err}")
