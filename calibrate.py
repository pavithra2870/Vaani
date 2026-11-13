"""Calibrate the stand-in from REAL gpt-oss-20b measurements.

Inputs  : runs/probe_openai_gpt-oss-20b.jsonl (probe stages) and every real 20b call logged in
          runs/eval_answers.jsonl / runs/calls_*-real.jsonl (per-call timelines)
Output  : runs/calibration.json  (standin_params are loaded by standin/server.py; the rest documents the fit)

What is fitted (and what is NOT):
  * TTFT        median = base + k * prompt_tokens (least squares on real calls), sigma from log-residuals
  * decode      ms per output token = (total - ttft) / output_tokens (median)
  * slow tail   share of calls whose TTFT > 3x the fitted median; their typical multiple
  * output size median output tokens per pipeline step
  * load slowdown from the probe's parallel stage (1 vs 8 calls in flight)
  * failures    NOT measured on the real service (0 errors seen): left as an assumption
"""
import glob
import json
import math
import statistics as st
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
MODEL = "openai/gpt-oss-20b"
ROOT = __import__("pathlib").Path(__file__).resolve().parent
RUNS = ROOT / "runs"


def real_calls():
    rows = []
    for l in open(RUNS / "probe_openai_gpt-oss-20b.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d["stage"] in ("latency", "tpm") and d["status"] == "ok" and d["ttft_ms"]:
            rows.append(dict(step="probe", tin=d["tin"], tout=d["tout"], ttft=d["ttft_ms"], total=d["total_ms"]))
    for f in glob.glob(str(RUNS / "eval_answers.jsonl")):
        for l in open(f, encoding="utf-8"):
            d = json.loads(l)
            if d["model"] != MODEL:
                continue
            for c in d["calls"]:
                if c["status"] == "ok" and c["ttft_ms"]:
                    rows.append(dict(step=c["step"], tin=c["tin"], tout=c["tout"], ttft=c["ttft_ms"], total=c["total_ms"]))
    for f in glob.glob(str(RUNS / "calls_*-real.jsonl")):
        for l in open(f, encoding="utf-8"):
            for c in json.loads(l)["calls"]:
                if c["status"] == "ok" and c["ttft_ms"]:
                    rows.append(dict(step=c["step"], tin=c["tin"], tout=c["tout"], ttft=c["ttft_ms"], total=c["total_ms"]))
    return rows


def main():
    rows = real_calls()
    tin = np.array([r["tin"] for r in rows], float)
    ttft = np.array([r["ttft"] for r in rows], float)
    # robust fit: least squares on the central 90% (the slow tail is modelled separately)
    k, b = np.polyfit(tin, ttft, 1)
    for _ in range(2):
        pred = b + k * tin
        keep = ttft < 2.5 * pred
        k, b = np.polyfit(tin[keep], ttft[keep], 1)
    b = max(b, 50.0)
    k = max(k, 0.0)
    pred = b + k * tin
    ratio = ttft / pred
    body = ratio[ratio < 2.5]
    sigma = float(np.std(np.log(body)))
    slow = ratio[ratio >= 2.5]
    slow_prob = float(len(slow) / len(ratio))
    slow_mult = float(np.median(slow)) if len(slow) else 4.0
    dec = [(r["total"] - r["ttft"]) / r["tout"] for r in rows if r["tout"] > 20]
    tout = {}
    for step in ("understand", "draft", "check"):
        v = [r["tout"] for r in rows if r["step"] == step]
        if v:
            tout[step] = int(st.median(v))
    # load slowdown from the probe's concurrency stage
    seq, par = [], []
    for l in open(RUNS / "probe_openai_gpt-oss-20b.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if d["stage"] == "concurrency" and d["status"] == "ok":
            n = int(d["tag"].split("-")[0][1:])
            (par if n == 8 else seq).append(d["total_ms"])
    load_alpha = max(0.0, (st.mean(par) / st.mean(seq) - 1) / 7) if par and seq else 0.0
    params = dict(ttft_base_ms=round(b, 1), ttft_per_in_token_ms=round(float(k), 4), ttft_sigma=round(sigma, 3),
                  decode_ms_per_token=round(float(st.median(dec)), 3) if dec else 1.0,
                  slow_prob=round(slow_prob, 4), slow_mult=round(slow_mult, 2),
                  load_alpha=round(load_alpha, 4), fail_prob=0.005,
                  rpm=30, tpm=8000, rpd=1000, tpd=200000, tout=tout)
    out = dict(model=MODEL, n_calls=len(rows), standin_params=params,
               notes=dict(fail_prob="ASSUMED - 0 errors observed in the real calls, so true rate is unknown",
                          load_alpha=f"from probe parallel=8 vs parallel<=4: mean total {st.mean(par):.0f} vs {st.mean(seq):.0f} ms",
                          daily_limit="429 text for RPD/TPD and its retry-after were never observed (assumed 3600 s)"))
    (RUNS / "calibration.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("ttft raw percentiles p50/p95/p99:", [round(float(np.percentile(ttft, q))) for q in (50, 95, 99)])


main()
