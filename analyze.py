"""Compute every table in the write-up from the raw files. Nothing is typed in by hand.

  python analyze.py            -> prints markdown, writes runs/tables.md

Sources: results.csv (+ results_detail.csv for detail/mode), runs/scenario_summaries.json (duration, offered rate,
peak queue), runs/probe_*.jsonl (Table 1 evidence).
Definitions (also stated in the write-up):
  * percentiles: numpy linear interpolation; latency percentiles are over requests with status == ok;
    first token = first token of the model's draft stream, so requests answered without a draft (refusals,
    busy / extractive messages) have no first token and are excluded from the first-token columns.
  * Succeeded % = status ok / sent.  Rate-limited % = status rate_limited (provider 429s we could not recover from,
    requests we turned away as busy, and daily-limit fallbacks).  Timed out or given up % = status timeout.
  * Tokens per answer = mean(tokens_in + tokens_out) over requests whose detail starts with 'answered'.
  * Cost per 1,000 answers = all tokens spent in the scenario (including requests that failed) priced at Groq's
    on-demand rates, divided by answers delivered, x 1000. Stand-in rows are priced as gpt-oss-20b.
"""
import csv
import json
import statistics as st
import sys
from collections import defaultdict
import numpy as np
from relay.config import ROOT

sys.stdout.reconfigure(encoding="utf-8")
PRICE = {  # USD per 1M tokens (input, output): https://console.groq.com/docs/models
    "openai/gpt-oss-20b": (0.075, 0.30), "openai/gpt-oss-120b": (0.15, 0.60), "stand-in": (0.075, 0.30)}


def pct(v, q):
    return float(np.percentile(v, q)) if v else None


def fmt(x, d=0):
    return "not measured" if x is None else f"{x:.{d}f}"


def triple(v):
    return "n/a (0 samples)" if not v else " / ".join(fmt(pct(v, q)) for q in (50, 95, 99))


def pair(v):
    return "n/a (0 samples)" if not v else " / ".join(fmt(pct(v, q)) for q in (50, 95))


def load():
    rows = list(csv.DictReader(open(ROOT / "results.csv", encoding="utf-8")))
    det = {d["request_id"]: d for d in csv.DictReader(open(ROOT / "results_detail.csv", encoding="utf-8"))}
    for r in rows:
        r.update({k: v for k, v in det[r["request_id"]].items() if k not in r})
        for k in ("total_ms", "tokens_in", "tokens_out", "retries"):
            r[k] = float(r[k])
        r["first_token_ms"] = float(r["first_token_ms"]) if r["first_token_ms"] else None
    return rows


def group(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["scenario"]].append(r)
    return g


def cost(rs):
    pin, pout = PRICE.get(rs[0]["model"], PRICE["stand-in"])
    usd = sum(r["tokens_in"] * pin + r["tokens_out"] * pout for r in rs) / 1e6
    answers = [r for r in rs if r["detail"].startswith("answered")]
    return (usd / len(answers) * 1000) if answers else None


def main():
    rows = load()
    g = group(rows)
    summ = json.loads((ROOT / "runs" / "scenario_summaries.json").read_text()) if (ROOT / "runs" / "scenario_summaries.json").exists() else {}
    out = []
    P = out.append
    load_names = [n for n in g if not n.startswith("eval-")]
    P("### Table 2a: latency per load scenario (ms; percentiles over status=ok requests)\n")
    P("| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |")
    P("|---|---|---|---|---|---|---|")
    for n in load_names:
        rs = g[n]
        sm = summ.get(n, {})
        ft = [r["first_token_ms"] for r in rs if r["status"] == "ok" and r["first_token_ms"] is not None]
        tt = [r["total_ms"] for r in rs if r["status"] == "ok"]
        P(f"| {n} | {rs[0]['source']} | {len(rs)} | {fmt(sm.get('duration_s'), 1)} | {fmt(sm.get('offered_per_min'), 1)} "
          f"| {triple(ft)} (n={len(ft)}) | {triple(tt)} (n={len(tt)}) |")
    P("\n### Table 2b: outcomes per load scenario\n")
    P("| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Error % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers (USD) |")
    P("|---|---|---|---|---|---|---|---|---|")
    for n in load_names:
        rs = g[n]
        N = len(rs)
        c = lambda s: 100 * sum(1 for r in rs if r["status"] == s) / N
        ans = [r for r in rs if r["detail"].startswith("answered")]
        tpa = st.mean(r["tokens_in"] + r["tokens_out"] for r in ans) if ans else None
        cp = cost(rs)
        P(f"| {n} | {c('ok'):.1f} | {c('rate_limited'):.1f} | {c('timeout'):.1f} | {c('error'):.1f} "
          f"| {st.mean(r['retries'] for r in rs):.2f} | {summ.get(n, {}).get('peak_queue', 'not measured')} "
          f"| {fmt(tpa)} (n={len(ans)}) | {'not measured' if cp is None else f'{cp:.2f}'} |")
    P("\n### Table 3 (and extras): evaluation runs, one row per setting / mode / model\n")
    P("| Scenario | Model | Mode | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Wrongly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in [n for n in g if n.startswith("eval-")]:
        rs = g[n]
        N = len(rs)
        gr = lambda s: 100 * sum(1 for r in rs if r["answer_grade"] == s) / N
        ft = [r["first_token_ms"] for r in rs if r["first_token_ms"] is not None]
        tt = [r["total_ms"] for r in rs]
        tpa = st.mean(r["tokens_in"] + r["tokens_out"] for r in rs)
        P(f"| {n} | {rs[0]['model']} | {rs[0]['mode']} | {N} | {gr('correct'):.1f} | {gr('wrong'):.1f} | {gr('refused_correctly'):.1f} "
          f"| {gr('refused_wrongly'):.1f} | {tpa:.0f} | {pair(ft)} | {pair(tt)} |")
    # ---- Table 3b: model-side first token (per successful draft call, from per-call logs) ----
    P("\n### Table 3b: model-side draft first token and call time per evaluation run (ms; per successful draft call, "
      "excludes queueing, pacing and retry waits)\n")
    P("| Scenario | Draft calls | Draft first token p50 / p95 | Draft call total p50 / p95 | Provider 429s hit |")
    P("|---|---|---|---|---|")
    ev = [json.loads(l) for l in open(ROOT / "runs" / "eval_answers.jsonl", encoding="utf-8")]
    by = defaultdict(list)
    for r in ev:
        by[r["scenario"]].append(r)
    for n, rs in by.items():
        d = [c for r in rs for c in r["calls"] if c["step"] == "draft" and c["status"] == "ok" and c["ttft_ms"]]
        n429 = sum(1 for r in rs for c in r["calls"] if c["status"] == "rate_limited")
        P(f"| {n} | {len(d)} | {pair([c['ttft_ms'] for c in d])} | {pair([c['total_ms'] for c in d])} | {n429} |")
    # ---- Table 3c: offline retrieval (no model): is a gold page among the retrieved pieces? ----
    off = ROOT / "runs" / "offline_retrieval.json"
    if off.exists():
        P("\n### Table 3c: offline retrieval quality (no model calls; raw question as query; 17 answerable questions)\n")
        P("| Setting | Gold page retrieved % | Avg words sent to the model |")
        P("|---|---|---|")
        for setting, rows_ in json.load(open(off)).items():
            a = [x for x in rows_ if x["gold_hit"] is not None]
            P(f"| {setting} | {100*sum(1 for x in a if x['gold_hit'])/len(a):.1f} | {st.mean(x['words'] for x in rows_):.0f} |")
    txt = "\n".join(out)
    (ROOT / "runs" / "tables.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
