"""Free (no model call) retrieval evaluation: for every setting, is the gold page in the retrieved set, and what is the
top BM25 score for answerable vs unanswerable questions? Used to pick RELEVANCE_MIN and to compare settings."""
import json
import sys
from relay import config
from relay.grade import gold_pages, load_questions
from relay.retrieve import retrieve

sys.stdout.reconfigure(encoding="utf-8")


def evaluate(setting, query_fn=lambda q: q["question"]):
    rows = []
    for q in load_questions():
        hits, top, cov = retrieve(query_fn(q), setting)
        got = {(h["paper"], h["page"]) for h in hits}
        ok = None
        if q["gold"]:
            ok = all(any((e["paper"], pg) in got for pg in gold_pages(e)) for e in q["gold"])
        words = sum(len(h["text"].split()) for h in hits)
        rows.append(dict(id=q["id"], type=q["type"], top=top, cov=cov, gold_hit=ok, words=words))
    return rows


if __name__ == "__main__":
    out = {}
    for s in config.RETRIEVAL_SETTINGS:
        rows = evaluate(s)
        out[s] = rows
        ans = [r for r in rows if r["gold_hit"] is not None]
        recall = sum(1 for r in ans if r["gold_hit"]) / len(ans)
        print(f"\n== {s}: gold-page recall {recall:.1%} ({sum(1 for r in ans if r['gold_hit'])}/{len(ans)}), "
              f"avg words sent {sum(r['words'] for r in rows)/len(rows):.0f}")
        print("  answerable top scores  :", sorted(round(r['top'], 1) for r in rows if r['type'] != 'unanswerable'))
        print("  unanswerable top scores:", sorted(round(r['top'], 1) for r in rows if r['type'] == 'unanswerable'))
        print("  missed:", [r["id"] for r in ans if not r["gold_hit"]])
    json.dump(out, open(config.RUNS_DIR / "offline_retrieval.json", "w"), indent=1)
