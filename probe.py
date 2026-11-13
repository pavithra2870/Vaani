"""Probe the REAL Groq service to fill Table 1 and calibrate the stand-in.
Every call is logged raw to runs/probe_<model>.jsonl (headers included).

  python probe.py --stage latency            # 10 sequential calls: latency spread, headers (cheap)
  python probe.py --stage concurrency        # 2/4/8 parallel tiny calls: any in-flight limit?
  python probe.py --stage limits             # burst of tiny calls until a 429: RPM, 429 format, recovery time
  python probe.py --stage tpm                # ~1.5K-token calls back to back until a TPM 429: how tokens are counted
  add --model openai/gpt-oss-120b to probe another model

Run one stage at a time and look at the output before the next. Uses your one account only.
"""
import argparse
import asyncio
import json
import time
import httpx
from relay import config, budget, llm

config.BACKEND = "groq"
PROMPT = ("Explain in two sentences what a rate limit is for an API and why a client should back off "
          "when it is hit. " * 3)


def logpath():
    config.RUNS_DIR.mkdir(exist_ok=True)
    return config.RUNS_DIR / f"probe_{config.GROQ_MODEL.replace('/', '_')}.jsonl"


async def one(client, stage, tag, max_tokens, prompt=PROMPT):
    if budget.exhausted():
        raise SystemExit("local daily cap reached - stop (see GROQ_DAILY_TOKEN_CAP)")
    t_wall = time.time()
    r = await llm.chat(client, [{"role": "user", "content": prompt}], "probe", max_tokens=max_tokens, timeout_s=60)
    budget.add(r.tokens_in + r.tokens_out)
    rec = dict(stage=stage, tag=tag, t=t_wall, model=config.GROQ_MODEL, status=r.status, http=r.http_status,
               ttft_ms=r.first_token_ms, total_ms=r.total_ms, tin=r.tokens_in, tout=r.tokens_out,
               retry_after=r.retry_after, headers=r.headers, error=r.error)
    with open(logpath(), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def show(rec):
    print(f"  {rec['tag']:>10} {rec['status']:<12} http={rec['http']} ttft={rec['ttft_ms'] and round(rec['ttft_ms'])} "
          f"total={round(rec['total_ms'])} tok={rec['tin']}+{rec['tout']} retry_after={rec['retry_after']}")


async def stage_latency(client):
    print("latency: 10 sequential calls")
    for i in range(10):
        rec = await one(client, "latency", f"seq{i}", 120)
        show(rec)
        await asyncio.sleep(2.5)    # stay far below RPM so we measure latency, not limits
    print("last headers:", json.dumps(rec["headers"]))


async def stage_concurrency(client):
    print("concurrency: parallel tiny calls")
    for n in (2, 4, 8):
        recs = await asyncio.gather(*[one(client, "concurrency", f"n{n}-{i}", 32, "Say OK.") for i in range(n)])
        print(f" parallel={n}: statuses={[r['status'] for r in recs]} "
              f"total_ms={[round(r['total_ms']) for r in recs]}")
        await asyncio.sleep(8)


async def stage_limits(client):
    print("limits: tiny calls back-to-back until the first 429 (max 45 calls)")
    first429 = None
    for i in range(45):
        rec = await one(client, "limits", f"b{i}", 16, "Say OK.")
        show(rec)
        if rec["status"] == "rate_limited":
            first429 = rec
            print("  429 headers:", json.dumps(rec["headers"]))
            print("  429 body:", rec["error"])
            break
    if not first429:
        print("no 429 in 45 calls: RPM is above 45 (or not the binding limit)")
        return
    print("recovery: one tiny call every 3s until success")
    t0 = time.time()
    while time.time() - t0 < 180:
        await asyncio.sleep(3)
        rec = await one(client, "limits", "recover", 16, "Say OK.")
        if rec["status"] == "ok":
            print(f"  recovered after {time.time() - t0:.1f}s")
            return
    print("  not recovered within 180s (likely a daily/other limit - read the 429 headers above)")


async def stage_tpm(client):
    import json as _j
    chunks = _j.loads((config.INDEX_DIR / "chunks_250_50.json").read_text(encoding="utf-8"))
    big = "Summarise in one sentence:\n" + "\n\n".join(c["text"] for c in chunks[40:46])
    print(f"tpm: ~{len(big) // 4} token prompts, max_tokens=200, back to back (max 7 calls)")
    for i in range(7):
        rec = await one(client, "tpm", f"t{i}", 200, big)
        show(rec)
        print("    remaining-tokens:", rec["headers"].get("x-ratelimit-remaining-tokens"),
              "reset-tokens:", rec["headers"].get("x-ratelimit-reset-tokens"))
        if rec["status"] == "rate_limited":
            print("  429 body:", rec["error"])
            break


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["latency", "concurrency", "limits", "tpm"])
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    if a.model:
        config.GROQ_MODEL = a.model
    if not config.GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY missing in .env")
    async with httpx.AsyncClient() as client:
        await {"latency": stage_latency, "concurrency": stage_concurrency, "limits": stage_limits,
                  "tpm": stage_tpm}[a.stage](client)
    print(f"logged to {logpath()} | local budget used today: {budget.used()} tokens")


asyncio.run(main())
