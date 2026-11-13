"""Stand-in model service (OpenAI-compatible /v1/chat/completions, streaming).

Behaviours (parameters are calibrated from the real Groq measurements by calibrate.py -> runs/calibration.json,
and can be changed at run time with POST /admin/config):
  * latency      : TTFT = lognormal(median = ttft_base_ms + ttft_per_in_token_ms * prompt_tokens, sigma)
                   then decode_ms_per_token per streamed output token
  * slow tail    : with prob slow_prob a call is slow_mult x slower (it SUCCEEDS slowly, it does not fail)
  * load slowdown: all latencies x (1 + load_alpha * (in_flight - 1))     [NOT measured on the real service]
  * failures     : with prob fail_prob -> HTTP 500
  * limits       : RPM = sliding 60 s window of requests; TPM = token bucket refilled continuously
                   (prompt tokens checked at admission, output tokens charged at completion - as observed on Groq);
                   RPD / TPD = day counters; max_concurrent = in-flight cap.
                   Over a limit -> HTTP 429, Groq-style message, retry-after, x-ratelimit-* headers.
  * streaming    : reply sent a few words at a time as SSE, with a final usage chunk.
Run:  python -m uvicorn standin.server:app --port 8001
"""
import asyncio
import json
import math
import random
import re
import time
from collections import deque
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

DEFAULTS = dict(
    ttft_base_ms=300.0, ttft_per_in_token_ms=0.0, ttft_sigma=0.4,
    decode_ms_per_token=1.0,
    slow_prob=0.03, slow_mult=4.0,
    load_alpha=0.05,
    fail_prob=0.01,
    tout={"understand": 40, "draft": 150, "check": 10},
    rpm=30, tpm=8000, rpd=1000, tpd=200000, max_concurrent=1000,
    seed=None,
)
CAL = Path(__file__).resolve().parent.parent / "runs" / "calibration.json"


def load_defaults():
    d = dict(DEFAULTS)
    if CAL.exists():
        d.update(json.loads(CAL.read_text()).get("standin_params", {}))
    return d


cfg = load_defaults()
rng = random.Random()
state = {}


def reset_state():
    state.clear()
    state.update(in_flight=0, peak_in_flight=0, req_log=deque(), day_req=0, day_tok=0,
                 tok_level=float(cfg["tpm"]), tok_t=time.time(),
                 counts=dict(ok=0, rate_limited=0, failed=0))


reset_state()


def est_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5))


def refill(now):
    rate = cfg["tpm"] / 60.0
    state["tok_level"] = min(float(cfg["tpm"]), state["tok_level"] + (now - state["tok_t"]) * rate)
    state["tok_t"] = now


def headers(retry_after=None):
    h = {"x-ratelimit-limit-requests": str(cfg["rpd"]),
         "x-ratelimit-remaining-requests": str(max(0, cfg["rpd"] - state["day_req"])),
         "x-ratelimit-limit-tokens": str(cfg["tpm"]),
         "x-ratelimit-remaining-tokens": str(max(0, int(state["tok_level"])))}
    if retry_after is not None:
        h["retry-after"] = str(max(1, math.ceil(retry_after)))
    return h


def limit_429(kind, retry_after, limit, used, requested):
    state["counts"]["rate_limited"] += 1
    unit = {"RPM": "requests per minute (RPM)", "TPM": "tokens per minute (TPM)",
            "RPD": "requests per day (RPD)", "TPD": "tokens per day (TPD)"}.get(kind, kind)
    msg = (f"Rate limit reached for model `standin` in organization `org_standin` service tier `on_demand` on {unit}: "
           f"Limit {limit}, Used {used}, Requested {requested}. Please try again in {retry_after:.1f}s.")
    return JSONResponse({"error": {"message": msg, "type": "tokens" if "T" in kind[:1] else "requests",
                                   "code": "rate_limit_exceeded"}},
                        status_code=429, headers=headers(retry_after))


def canned(step: str, prompt: str, target_tokens: int) -> str:
    if step == "understand":
        q = prompt.strip().split("\n")[-1]
        return json.dumps({"query": q, "in_scope": True})
    if step == "check":
        return "PASS"
    m = re.search(r"\[([a-z]+ p\d+)\]", prompt.split("Passages:", 1)[-1])
    label = m.group(1) if m else "attention p1"
    text = (f"According to the retrieved passage, this is addressed in the paper [{label}]. ")
    filler = "The mechanism is described there in detail and supported by the reported experiments. "
    while est_tokens(text) < target_tokens - 12:
        text += filler
    return text + f"[{label}]"


@app.post("/admin/config")
async def set_config(request: Request):
    body = await request.json()
    cfg.update(body)
    if "tpm" in body:
        state["tok_level"] = min(state["tok_level"], float(cfg["tpm"]))
    if cfg.get("seed") is not None:
        rng.seed(cfg["seed"])
    return cfg


@app.post("/admin/reset")
async def reset(request: Request):
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    cfg.clear(); cfg.update(load_defaults()); cfg.update(body)
    if cfg.get("seed") is not None:
        rng.seed(cfg["seed"])
    reset_state()
    return cfg


@app.get("/admin/stats")
async def stats():
    return {"cfg": cfg, "in_flight": state["in_flight"], "peak_in_flight": state["peak_in_flight"],
            "day_req": state["day_req"], "day_tok": state["day_tok"], "counts": state["counts"]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    now = time.time()
    body = await request.json()
    step = (body.get("metadata") or {}).get("step", "draft")
    prompt = "\n".join(m["content"] for m in body.get("messages", []))
    tin = est_tokens(prompt)

    # ---- limits (checked in the order Groq reports them) ----
    log = state["req_log"]
    while log and now - log[0] >= 60:
        log.popleft()
    refill(now)
    if state["day_req"] >= cfg["rpd"]:
        return limit_429("RPD", 3600.0, cfg["rpd"], state["day_req"], 1)
    if state["day_tok"] + tin > cfg["tpd"]:
        return limit_429("TPD", 3600.0, cfg["tpd"], state["day_tok"], tin)
    if state["in_flight"] >= cfg["max_concurrent"]:
        return limit_429("RPM", 1.0, cfg["rpm"], len(log), 1)
    if len(log) >= cfg["rpm"]:
        return limit_429("RPM", max(0.5, 60 - (now - log[0])), cfg["rpm"], len(log), 1)
    if state["tok_level"] < tin:
        return limit_429("TPM", (tin - state["tok_level"]) / (cfg["tpm"] / 60.0), cfg["tpm"],
                         int(cfg["tpm"] - state["tok_level"]), tin)

    log.append(now)
    state["tok_level"] -= tin
    state["day_req"] += 1
    state["in_flight"] += 1
    state["peak_in_flight"] = max(state["peak_in_flight"], state["in_flight"])
    load_mult = 1 + cfg["load_alpha"] * (state["in_flight"] - 1)

    if rng.random() < cfg["fail_prob"]:
        await asyncio.sleep(0.05)
        state["in_flight"] -= 1
        state["counts"]["failed"] += 1
        return JSONResponse({"error": {"message": "internal error"}}, status_code=500, headers=headers())

    mult = load_mult * (cfg["slow_mult"] if rng.random() < cfg["slow_prob"] else 1.0)
    median = cfg["ttft_base_ms"] + cfg["ttft_per_in_token_ms"] * tin
    ttft = rng.lognormvariate(math.log(median), cfg["ttft_sigma"]) * mult / 1000
    target = int(max(1, rng.gauss(cfg["tout"].get(step, 100), cfg["tout"].get(step, 100) * 0.15)))
    target = min(target, body.get("max_tokens", 300))
    text = canned(step, prompt, target)
    tout = est_tokens(text) if step != "check" else target
    hdrs = headers()

    async def gen():
        try:
            await asyncio.sleep(ttft)
            words = text.split(" ")
            per_word = cfg["decode_ms_per_token"] * (tout / max(1, len(words))) * mult / 1000
            for i in range(0, len(words), 3):
                piece = " ".join(words[i:i + 3]) + " "
                yield f"data: {json.dumps({'choices': [{'delta': {'content': piece}}]})}\n\n"
                await asyncio.sleep(per_word * 3)
            state["tok_level"] -= tout            # output tokens are charged at completion
            state["day_tok"] += tin + tout
            yield f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': tin, 'completion_tokens': tout}})}\n\n"
            yield "data: [DONE]\n\n"
            state["counts"]["ok"] += 1
        finally:
            state["in_flight"] -= 1

    return StreamingResponse(gen(), media_type="text/event-stream", headers=hdrs)
