"""Single source of truth for where things live on disk.

Every module imports its directories from here instead of computing them from
``__file__``. That was the thing that quietly broke whenever a file moved, and
it is the reason the layout can be reorganised without touching logic.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CACHE = ROOT / "data_cache"      # market data, trained models (gitignored)
RUNTIME = ROOT / "runtime"       # live logs written while the server runs
WEB = ROOT / "oracle" / "app" / "web"
DOCS = ROOT / "docs"
RESEARCH = ROOT / "research"
LOGS = ROOT / "logs"

for _d in (CACHE, RUNTIME, LOGS):
    _d.mkdir(parents=True, exist_ok=True)
