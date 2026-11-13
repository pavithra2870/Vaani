"""One streaming chat call (a single attempt, no retries - retries live in gateway.py).

Talks to either Groq or the stand-in; both speak the OpenAI chat-completions protocol.
Measures time-to-first-token (first content delta) and total time.
"""
import hashlib
import json
import time
import httpx
from . import config


class LLMResult:
    __slots__ = ("status", "text", "first_token_ms", "total_ms", "tokens_in", "tokens_out",
                 "http_status", "retry_after", "headers", "error", "cached")

    def __init__(self):
        self.status = "error"        # ok | rate_limited | timeout | error
        self.text = ""
        self.first_token_ms = None
        self.total_ms = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.http_status = None
        self.retry_after = None      # seconds, from retry-after header on 429
        self.headers = {}
        self.error = ""
        self.cached = False


def _target():
    if config.BACKEND == "groq":
        return config.GROQ_URL, {"Authorization": f"Bearer {config.GROQ_API_KEY}"}
    return config.STANDIN_URL, {}


def _cache_path(payload):
    config.CACHE_DIR.mkdir(exist_ok=True)
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return config.CACHE_DIR / f"{h}.json"


def _keep_headers(h):
    keep = {}
    for k, v in h.items():
        k = k.lower()
        if k.startswith("x-ratelimit") or k == "retry-after":
            keep[k] = v
    return keep


async def chat(client: httpx.AsyncClient, messages, step: str, max_tokens=400, timeout_s=30.0) -> LLMResult:
    url, auth = _target()
    payload = {
        "model": config.GROQ_MODEL if config.BACKEND == "groq" else "standin",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "metadata": {"step": step},  # used by the stand-in to choose a canned reply shape
    }
    if config.BACKEND == "groq":
        payload.pop("metadata")
        payload["reasoning_effort"] = "low"   # gpt-oss: keep hidden reasoning short (it costs tokens + time)

    res = LLMResult()
    cp = _cache_path(payload) if (config.LLM_CACHE and config.BACKEND == "groq") else None
    if cp and cp.exists():
        d = json.loads(cp.read_text())
        for k, v in d.items():
            setattr(res, k, v)
        res.cached = True
        return res

    t0 = time.perf_counter()
    try:
        async with client.stream("POST", url, json=payload, headers=auth, timeout=timeout_s) as r:
            res.http_status = r.status_code
            res.headers = _keep_headers(r.headers)
            if r.status_code == 429:
                res.status = "rate_limited"
                ra = r.headers.get("retry-after")
                res.retry_after = float(ra) if ra else None
                res.error = "429 " + (await r.aread()).decode("utf-8", "ignore")[:400]
            elif r.status_code != 200:
                res.status = "error"
                body = (await r.aread()).decode("utf-8", "ignore")[:200]
                res.error = f"{r.status_code} {body}"
            else:
                parts = []
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    usage = obj.get("usage") or (obj.get("x_groq") or {}).get("usage")
                    if usage:
                        res.tokens_in = usage.get("prompt_tokens", 0)
                        res.tokens_out = usage.get("completion_tokens", 0)
                    for ch in obj.get("choices", []):
                        piece = (ch.get("delta") or {}).get("content")
                        if piece:
                            if res.first_token_ms is None:
                                res.first_token_ms = (time.perf_counter() - t0) * 1000
                            parts.append(piece)
                res.text = "".join(parts)
                res.status = "ok"
    except (httpx.TimeoutException, httpx.ReadError) as e:
        res.status = "timeout"
        res.error = type(e).__name__
    except httpx.HTTPError as e:
        res.status = "error"
        res.error = type(e).__name__
    res.total_ms = (time.perf_counter() - t0) * 1000

    if cp and res.status == "ok":
        cp.write_text(json.dumps({k: getattr(res, k) for k in LLMResult.__slots__ if k != "cached"}))
    return res
