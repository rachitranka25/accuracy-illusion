#!/usr/bin/env python3
"""
Credential loading — environment variables (.env) first, JSON-file fallback.

Why: so the project can live in a shared Git repo with secrets kept OUT of git.
Each machine keeps its own secrets in a local `.env` (gitignored) OR the old
`*_config.json` (also gitignored). Only `.env.example` is committed, so teammates
know WHICH keys are needed without ever seeing the values.

No external dependency — a tiny .env parser is built in.
"""
import os
import json
from pathlib import Path

from oracle.paths import ROOT as HERE


def load_dotenv(path=None):
    """Read a KEY=value `.env` file into os.environ (real env vars still win)."""
    f = Path(path) if path else HERE / ".env"
    if not f.exists():
        return
    for raw in f.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


load_dotenv()   # load once on import


def _valid(v):
    return bool(str(v).strip()) and "FILL" not in str(v)


def from_env_or_json(json_name, env_map, require_all=True):
    """Build a config dict.

    env_map: {config_key: ENV_VAR_NAME}. For each key, an environment variable
    wins if set; otherwise the value comes from `json_name`. Extra keys present
    only in the JSON (e.g. a broker's daily access_token) are carried through.

    Returns None if `require_all` and any mapped key is still missing.
    """
    file_cfg = {}
    f = HERE / json_name
    if f.exists():
        try:
            file_cfg = json.loads(f.read_text())
        except Exception:
            file_cfg = {}

    out = {}
    for key, env_var in env_map.items():
        env_val = os.environ.get(env_var, "")
        out[key] = env_val if _valid(env_val) else file_cfg.get(key, "")
    for k, v in file_cfg.items():        # carry extra JSON-only keys
        out.setdefault(k, v)

    if require_all and not all(_valid(out.get(k)) for k in env_map):
        return None
    return out
