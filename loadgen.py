"""Load generator: pretend to be many users. Arrivals are OPEN-LOOP (a request arrives on schedule whether or not
earlier ones have finished), which is what a real burst looks like.

  python loadgen.py --list
  python loadgen.py --scenario steady-sim                      # stand-in (starts it automatically)
  python loadgen.py --scenario steady-real --model openai/gpt-oss-20b   # REAL Groq (spends your quota!)

Every request -> results.csv (+ results_detail.csv). One summary row per scenario -> runs/scenario_summaries.json.
Questions are drawn from eval/questions.json (seeded shuffle, cycled), so load rows are also graded.
"""
import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
import time
import httpx
from relay import config, results, budget
from relay.gateway import Gateway, GatewayConfig
from relay.grade import grade, load_questions
from relay import pipeline

STANDIN_PORT = 8001


def steady(n, per_min):
    gap = 60.0 / per_min
    return [i * gap for i in range(n)]


def poisson(duration_s, per_min, seed):
    r = random.Random(seed)
    t, out = 0.0, []
    while True:
        t += r.expovariate(per_min / 60.0)
        if t > duration_s:
            return out
        out.append(t)


def burst(n, window_s, tail_at=(60, 120, 180)):
    return [i * window_s / n for i in range(n)] + list(tail_at)


# standin: parameter overrides for the stand-in; gw: GatewayConfig overrides
SCENARIOS = {
    "dev-sim": dict(source="stand-in", arrivals=burst(10, 1, tail_at=(8,)), standin={}, gw={}),   # smoke test only
    # ---- REAL Groq (small: the free tier cannot take more) ----
    "steady-real":  dict(source="real", arrivals=steady(6, 2), standin={}, gw={}),
    "burst-real":   dict(source="real", arrivals=burst(6, 1, tail_at=(45, 90)), standin={}, gw={}),
    # ---- stand-in (calibrated to the real limits) ----
    "steady-sim":   dict(source="stand-in", arrivals=steady(30, 3), standin={}, gw={}),
    "burst-sim":    dict(source="stand-in", arrivals=burst(40, 3), standin={}, gw={}),
    "overload-sim": dict(source="stand-in", arrivals=poisson(300, 12, seed=7), standin={}, gw={}),
    "limit-halved-sim": dict(source="stand-in", arrivals=steady(18, 3), standin=dict(rpm=15, tpm=4000),
                             gw=dict(rpm_limit=30, tpm_limit=8000)),      # gateway still believes the OLD limits
    "daily-exhausted-sim": dict(source="stand-in", arrivals=steady(24, 6), standin=dict(rpd=15), gw={}),
    # ---- retry case study: same overload, NO pacing and NO admission control; immediate retries vs no retries ----
    "overload-naive-retry-sim": dict(source="stand-in", arrivals=poisson(300, 12, seed=7), standin={},
                                     gw=dict(retry_mode="naive", pace=False, admission=False, max_queue=10_000)),
    "overload-no-retry-sim": dict(source="stand-in", arrivals=poisson(300, 12, seed=7), standin={},
                                  gw=dict(max_retries=0, pace=False, admission=False, max_queue=10_000)),
}


def start_standin():
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "standin.server:app", "--port", str(STANDIN_PORT),
                          "--log-level", "warning"], cwd=str(config.ROOT))
    for _ in range(50):
        try:
            httpx.get(f"http://127.0.0.1:{STANDIN_PORT}/admin/stats", timeout=1)
            return p
        except Exception:
            time.sleep(0.2)
    p.kill()
    raise RuntimeError("stand-in did not start")


async def run_scenario(name, spec, model):
    standin = spec["source"] == "stand-in"
    config.BACKEND = "standin" if standin else "groq"
    if not standin and model:
        config.GROQ_MODEL = model
    proc = None
    if standin:
        proc = start_standin()
        httpx.post(f"http://127.0.0.1:{STANDIN_PORT}/admin/reset", json=dict(spec["standin"], seed=1))
    try:
        gw = Gateway(GatewayConfig(**spec["gw"]))
        qs = load_questions()
        order = random.Random(42).sample(range(len(qs)), len(qs))
        arrivals = spec["arrivals"]
        setting = config.DEFAULT_SETTING
        n = len(arrivals)
        print(f"[{name}] {n} requests over {max(arrivals):.0f}s ({spec['source']}) setting={setting}")
        t_zero = time.perf_counter()
        wall_zero = time.time()
        ends = []

        async def one(i, at):
            await asyncio.sleep(max(0, at - (time.perf_counter() - t_zero)))
            q = qs[order[i % len(qs)]]
            t0 = time.time()
            r = await pipeline.run(gw, q["question"], setting, "auto", deadline_s=gw.cfg.deadline_s)
            # stand-in answers are canned text, so only REAL answers are graded
            g = grade(q, r.answer, r.detail, r.citations, r.status) if not standin else ""
            ends.append(time.perf_counter() - t_zero)
            results.record(f"{name}-{i:03d}", name, spec["source"], t0, r, g, q["id"],
                           config.GROQ_MODEL if not standin else "stand-in", gw.waiting, t0 - wall_zero)
            with open(config.RUNS_DIR / f"calls_{name}.jsonl", "a", encoding="utf-8") as cf:
                cf.write(json.dumps(dict(request_id=f"{name}-{i:03d}", question_id=q["id"], status=r.status,
                                         detail=r.detail, mode=r.mode, total_ms=r.total_ms,
                                         first_token_ms=r.ctx.first_token_ms, queue_ms=r.ctx.queue_ms,
                                         work_ms=r.ctx.work_ms, retry_wait_ms=r.ctx.retry_wait_ms,
                                         calls=r.ctx.calls)) + "\n")
            print(f"  {i:3d} t={at:6.1f} {q['id']:>3} {r.status:<12} {r.detail:<28} mode={r.mode:<4} "
                  f"total={r.total_ms:7.0f} retries={r.ctx.retries} {g}")

        await asyncio.gather(*[one(i, a) for i, a in enumerate(arrivals)])
        duration = max(ends)
        span = max(arrivals) - min(arrivals)
        summary = dict(scenario=name, source=spec["source"], requests=n, duration_s=round(duration, 1),
                       offered_per_min=round((n - 1) / span * 60, 1) if span > 0 else None,
                       peak_queue=gw.peak_queue, setting=setting,
                       model=config.GROQ_MODEL if not standin else "stand-in",
                       standin_overrides=spec["standin"], gateway_overrides=spec["gw"],
                       standin_stats=(httpx.get(f"http://127.0.0.1:{STANDIN_PORT}/admin/stats").json()
                                      if standin else None))
        p = config.RUNS_DIR / "scenario_summaries.json"
        allsum = json.loads(p.read_text()) if p.exists() else {}
        allsum[name] = summary
        p.write_text(json.dumps(allsum, indent=1))
        await gw.close()
        print(f"[{name}] done in {duration:.0f}s, peak queue {gw.peak_queue}")
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario")
    ap.add_argument("--model", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.scenario:
        for k, v in SCENARIOS.items():
            print(f"{k:28} {v['source']:9} {len(v['arrivals'])} req")
        sys.exit(0)
    asyncio.run(run_scenario(a.scenario, SCENARIOS[a.scenario], a.model))
