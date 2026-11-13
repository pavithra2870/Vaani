"""Automatic grader for the evaluation questions (eval/questions.json).

Grades (the brief's four): correct | wrong | refused_correctly | refused_wrongly   ('' = no answer was delivered)
  answerable   : refusal            -> refused_wrongly
                 answer             -> correct only if (a) every key-fact group appears in the answer text AND
                                       (b) the answer cites a page that contains the gold fact, for EVERY gold paper
                                       (cross-paper questions need both papers); otherwise wrong
  unanswerable : refusal / "the papers do not say"  -> refused_correctly ; anything else -> wrong
Keyword matching is crude, so eval.py also keeps every answer text; overrides live in eval/grade_overrides.json.
"""
import json
import re
from functools import lru_cache
from .config import DATA_DIR, PAPERS
from .ingest import read_pages
from .pipeline import REFUSAL

REFUSAL_RE = re.compile(r"(papers? (in this library )?(do(es)? not|don.t) (say|state|specify|mention|report|provide)"
                        r"|not (stated|specified|mentioned|reported|provided|given) in the (papers|passages)"
                        r"|(passages|papers) (do(es)? not|don.t) (contain|include|say|state|specify))", re.I)


@lru_cache(maxsize=None)
def pages_text(paper):
    return {pg: txt for pg, txt in read_pages(DATA_DIR / PAPERS[paper]["file"])}


def gold_pages(entry):
    rx = re.compile(entry["page_regex"], re.I)
    return {pg for pg, txt in pages_text(entry["paper"]).items() if rx.search(txt)}


def normalize(t):
    """gpt-oss writes non-breaking hyphens/spaces ('bits‑per‑byte', '500 k'); fold them before matching."""
    t = re.sub(r"[‐-―−]", "-", t)
    t = re.sub(r"[    ]", " ", t)
    return t.lower()


def is_refusal(answer, detail):
    return detail.startswith("refused") or answer.strip().startswith(REFUSAL[:30]) or bool(REFUSAL_RE.search(answer))


def grade(q, answer, detail, citations, status):
    if status != "ok" or detail in ("shed_busy", "budget_exhausted_extractive"):
        return ""
    refused = is_refusal(answer, detail)
    if q["type"] == "unanswerable":
        return "refused_correctly" if refused else "wrong"
    if refused:
        return "refused_wrongly"
    text = normalize(answer)
    keys_ok = all(any(normalize(alt) in text for alt in group) for group in q["keys"])
    cited = {(p, int(pg)) for p, pg in citations}
    cite_ok = all(any((e["paper"], pg) in cited for pg in gold_pages(e)) for e in q["gold"])
    return "correct" if (keys_ok and cite_ok) else "wrong"


def load_questions(path=None):
    from .config import ROOT
    return json.loads((path or ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    for q in load_questions():
        for e in q["gold"]:
            print(q["id"], e["paper"], sorted(gold_pages(e)) or "NO MATCH !!!")
