"""Run the evaluation questions through the pipeline and grade them.

  python -m eval.run --backend groq --model openai/gpt-oss-120b --setting medium-k4 --mode lean
Writes rows to results.csv (scenario = eval-<setting>-<mode>) and full answers to runs/eval_answers.jsonl.
"""
import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from relay import config, budget, results
from relay.gateway import Gateway, GatewayConfig
from relay.grade import grade, load_questions
from relay import pipeline


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="groq")
    ap.add_argument("--model", default=None)
    ap.add_argument("--setting", required=True)
    ap.add_argument("--mode", default="lean", choices=["lean", "full"])
    ap.add_argument("--only", default=None, help="comma-separated question ids")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    config.BACKEND = a.backend
    if a.model:
        config.GROQ_MODEL = a.model
    source = "real" if a.backend == "groq" else "stand-in"
    qs = load_questions()
    if a.only:
        keep = set(a.only.split(","))
        qs = [q for q in qs if q["id"] in keep]
    scenario = f"eval-{a.setting}-{a.mode}{a.tag}"
    gw = Gateway(GatewayConfig(max_concurrency=2, deadline_s=180, call_timeout_s=60, max_queue=10_000))
    runs = config.RUNS_DIR
    runs.mkdir(exist_ok=True)
    for i, q in enumerate(qs):
        if budget.exhausted():
            print("local daily cap reached - stopping"); break
        t0 = time.time()
        r = await pipeline.run(gw, q["question"], a.setting, a.mode, deadline_s=180)
        g = grade(q, r.answer, r.detail, r.citations, r.status)
        results.record(f"{scenario}-{i:03d}", scenario, source, t0, r, g, q["id"], config.GROQ_MODEL if a.backend == "groq" else "stand-in")
        with open(Path(os.getenv("ANSWERS_FILE", runs / "eval_answers.jsonl")), "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(scenario=scenario, model=config.GROQ_MODEL, qid=q["id"], question=q["question"],
                                    answer=r.answer, detail=r.detail, status=r.status, citations=r.citations,
                                    retrieved=r.retrieved, grade=g, tokens_in=r.ctx.tokens_in,
                                    tokens_out=r.ctx.tokens_out, first_token_ms=r.ctx.first_token_ms,
                                    total_ms=r.total_ms, calls=r.ctx.calls), ensure_ascii=False) + "\n")
        print(f"{q['id']:>3} {g or '-':<18} {r.detail:<22} tok={r.ctx.tokens_in}+{r.ctx.tokens_out} "
              f"ttft={r.ctx.first_token_ms and round(r.ctx.first_token_ms)} total={round(r.total_ms)} "
              f"queue={round(r.ctx.queue_ms)} retries={r.ctx.retries}")
    await gw.close()
    print("budget used today (", config.GROQ_MODEL, "):", budget.used())


asyncio.run(main())
