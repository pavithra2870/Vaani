"""Re-grade saved answers (runs/eval_answers.jsonl) with the current grader and rewrite answer_grade in results.csv.
Used after the grader was corrected for false negatives found by reading the answers; the same grader is applied to
every setting. Also writes runs/regrade_changes.txt listing every grade that changed."""
import csv
import json
import sys
from collections import defaultdict
from relay.config import ROOT
from relay.grade import grade, load_questions

sys.stdout.reconfigure(encoding="utf-8")
qs = {q["id"]: q for q in load_questions()}
rows = [json.loads(l) for l in open(ROOT / "runs" / "eval_answers.jsonl", encoding="utf-8")]
idx = defaultdict(int)
new = {}
changes = []
for r in rows:
    i = idx[r["scenario"]]
    idx[r["scenario"]] += 1
    rid = f"{r['scenario']}-{i:03d}"
    g = grade(qs[r["qid"]], r["answer"], r["detail"], r["citations"], r["status"])
    if g != r["grade"]:
        changes.append(f"{rid} {r['qid']}: {r['grade'] or '-'} -> {g or '-'}")
        r["grade"] = g
    new[rid] = g
with open(ROOT / "runs" / "eval_answers.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
src = list(csv.DictReader(open(ROOT / "results.csv", encoding="utf-8")))
cols = list(src[0].keys())
for r in src:
    if r["request_id"] in new:
        r["answer_grade"] = new[r["request_id"]]
with open(ROOT / "results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(src)
(ROOT / "runs" / "regrade_changes.txt").write_text("\n".join(changes), encoding="utf-8")
print(len(changes), "grades changed")
print("\n".join(changes))
