"""Self-imposed daily token budget for the REAL Groq account, tracked PER MODEL (Groq limits are per model).
Persisted on disk, resets per UTC calendar day. This is a guard, not Groq's own counter: Groq's real counters
come back in x-ratelimit-* headers and 429 messages, which we log too.
"""
import json
import datetime as dt
from . import config


def _path():
    config.CACHE_DIR.mkdir(exist_ok=True)
    return config.CACHE_DIR / f"budget_{dt.datetime.now(dt.timezone.utc).date()}.json"


def _load():
    p = _path()
    return json.loads(p.read_text()) if p.exists() else {}


def used(model: str | None = None) -> int:
    return _load().get(model or config.GROQ_MODEL, 0)


def add(tokens: int, model: str | None = None):
    d = _load()
    m = model or config.GROQ_MODEL
    d[m] = d.get(m, 0) + tokens
    _path().write_text(json.dumps(d))


def exhausted() -> bool:
    return config.BACKEND == "groq" and used() >= config.GROQ_DAILY_TOKEN_CAP
