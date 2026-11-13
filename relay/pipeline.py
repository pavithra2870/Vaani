"""The Relay pipeline. Five steps, one tool call, one branch, one loop:

  1 understand  (model call)  - rewrite the question into a search query, flag obviously off-topic ones
  2 retrieve    (TOOL call)   - BM25 over the papers; has its own timeout
  3 decide      (BRANCH)      - weak evidence -> refuse WITHOUT spending a draft/check call
  4 draft       (model call)  - cited answer (streamed from the model; we BUFFER it until it is checked)
  5 check       (model call + code, LOOP) - citations must point at retrieved passages and (full mode) a
                                model must confirm the claims are supported; one redraft, then refuse.

Modes
  full : understand + draft + check            (3 model calls)
  lean : draft + code-only citation check      (1 model call)  <- weaker: nobody verifies the claims
  auto : full when the token budget has room, lean when pressure is high  (what the load tests use)

Degraded outcomes (all explained to the user, none silent):
  shed_busy                    too many waiting; "try again in N s"
  budget_exhausted_extractive  no model calls possible (daily limit); show the top retrieved passages as-is
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass
from . import budget, config
from .gateway import Gateway, ReqCtx
from .retrieve import retrieve

REFUSAL = "The papers in this library do not say. I can't answer that from them."
# gpt-oss writes citations like 【attention p5】 (full-width brackets, narrow no-break spaces) as well as [attention p5];
# several sources may share one bracket: [attention p4, p5; moshi p3]
GROUP_RE = re.compile(r"[\[\(【][^\]\)】]*[\]\)】]")
PAPER_PAGE_RE = re.compile(r"(attention|moshi|dynchunk)|p\.?\s*(\d+)", re.I)

UNDERSTAND_SYS = (
    "You prepare search queries over three papers: 'Attention Is All You Need' (Transformer), "
    "'Moshi' (speech-text dialogue model), 'Dynamic Chunking' (H-Net, hierarchical sequence modeling). "
    'Reply with ONLY JSON: {"query": "<keywords for searching the papers>", "in_scope": true|false}. '
    "in_scope=false only if the question clearly has nothing to do with these papers."
)
DRAFT_SYS = (
    "Answer the question using ONLY the passages given. After every claim cite its source as [paper pN], "
    "e.g. [attention p4], using the exact labels of the passages. Be concise (max 120 words). "
    "If the passages do not contain the answer, reply exactly: " + REFUSAL
)
CHECK_SYS = (
    "You verify an answer against passages. Reply with ONLY 'PASS' if every claim in the answer is "
    "supported by the passages, else 'FAIL: <short reason>'."
)


@dataclass
class Result:
    answer: str
    status: str            # ok | rate_limited | timeout | error   (what the CSV records)
    detail: str            # answered | answered_unverified | refused_* | shed_busy | budget_exhausted_extractive | ...
    citations: list
    setting: str
    mode: str              # mode actually used: full | lean | none
    ctx: ReqCtx
    total_ms: float
    retrieved: list        # [(paper, page)] of the passages shown to the model


def _label(c):
    return f"{c['paper']} p{c['page']}"


def _passages(hits):
    return "\n\n".join(f"[{_label(h)}]\n{h['text']}" for h in hits)


def _norm_cites(text):
    text = re.sub(r"[    ]", " ", text)
    out = []
    for g in GROUP_RE.findall(text):
        paper = None
        for m in PAPER_PAGE_RE.finditer(g):
            if m.group(1):
                paper = m.group(1).lower()
            elif paper:
                out.append((paper, int(m.group(2))))
    return out


def _cites_valid(text, hits):
    cites = _norm_cites(text)
    have = {(h["paper"], h["page"]) for h in hits}
    return bool(cites) and all(c in have for c in cites)


def _mins(sec):
    sec = max(1, int(round(sec)))
    return f"{sec} s" if sec < 90 else f"about {round(sec / 60)} min"


def _weak(top, cov):
    return top < config.RELEVANCE_MIN or cov < config.COVERAGE_MIN


def _extractive(hits):
    return "\n".join(f"- [{_label(h)}] {h['text'][:350]}..." for h in hits[:3])


def _fail_status(r):
    return "rate_limited" if r.status in ("rate_limited", "budget_exhausted") else r.status


def _fail_msg(ctx):
    hint = ctx.retry_after_hint_s
    if hint:
        return f"The service is at its limit. Please try again in {_mins(hint)}."
    return "Sorry, this took too long and I stopped working on it. Please try again."


async def run(gw: Gateway, question: str, setting: str = config.DEFAULT_SETTING,
              mode: str = "auto", deadline_s: float | None = None) -> Result:
    ctx = ReqCtx(deadline_s=deadline_s or gw.cfg.deadline_s)
    try:
        return await _run(gw, ctx, question, setting, mode)
    finally:
        gw.release(ctx)


async def _run(gw: Gateway, ctx: ReqCtx, question: str, setting: str, mode: str) -> Result:
    used_mode = "none"
    retrieved = []

    def done(answer, status, detail, cites=()):
        return Result(answer, status, detail, list(cites), setting, used_mode, ctx,
                      (time.perf_counter() - ctx.t_start) * 1000, list(retrieved))

    async def retrieve_hits(q):
        return await asyncio.wait_for(asyncio.to_thread(retrieve, q, setting), timeout=2.0)

    async def extractive_fallback(hits_top=None):
        """No model call possible (daily limit): show the best passages, unedited, and say so."""
        if hits_top is None:
            try:
                hits_top = await retrieve_hits(question)
            except asyncio.TimeoutError:
                return done("", "timeout", "retrieval tool timeout")
        h, top, cov = hits_top
        retrieved.extend((x["paper"], x["page"]) for x in h)
        blocked = gw.daily_blocked()
        when = _mins(blocked) if blocked > 0 else "later (the daily budget refills gradually)"
        ctx.retry_after_hint_s = blocked or None
        msg = ("I've used up today's model budget, so I can't write an answer right now. "
               "Here are the most relevant passages from the papers, unedited:\n"
               + (_extractive(h) if h and not _weak(top, cov) else "(nothing relevant found)")
               + f"\nPlease ask again in {when} for a written, checked answer.")
        return done(msg, "rate_limited", "budget_exhausted_extractive")

    # --- admission: daily limit first, then the token-budget admission control picks full / lean / busy ---
    if gw.daily_blocked() > 0 or budget.exhausted():
        return await extractive_fallback()

    if mode != "auto" or not gw.cfg.admission:
        used_mode = "full" if mode == "auto" else mode           # no protection: used for baselines and evaluation
    else:
        used_mode, retry_in = gw.admit(ctx, mode)
        if used_mode is None:
            used_mode = "none"
            return done("I'm handling a lot of questions right now and can't take yours without making you wait too "
                        f"long. Please try again in {_mins(retry_in)}.", "rate_limited", "shed_busy")

    # 1 understand
    query = question
    if used_mode == "full":
        r = await gw.call(ctx, [{"role": "system", "content": UNDERSTAND_SYS},
                                {"role": "user", "content": question}], "understand", max_tokens=150)
        if r.status == "budget_exhausted":
            return await extractive_fallback()
        if r.status != "ok":
            return done(_fail_msg(ctx), _fail_status(r), r.error or "understand failed")
        try:
            j = json.loads(re.search(r"\{.*\}", r.text, re.S).group(0))
            query = j.get("query") or question
            # NOTE: the model's in_scope flag is deliberately NOT used to refuse. In the first full-mode evaluation it
            # wrongly called 3 answerable questions off-topic (and caught 1 real one); retrieval evidence gates instead.
        except Exception:
            query = question   # model didn't give JSON: fall back to the raw question

    # 2 retrieve (tool call, with its own timeout)
    try:
        hits, top, cov = await retrieve_hits(query)
    except asyncio.TimeoutError:
        return done("", "timeout", "retrieval tool timeout")
    retrieved.extend((h["paper"], h["page"]) for h in hits)

    # 3 decide (branch)
    if not hits or _weak(top, cov):
        return done(REFUSAL, "ok", "refused_weak_evidence")

    # 4 draft  (+ 5 check loop)
    feedback = ""
    for attempt in range(2):
        msgs = [{"role": "system", "content": DRAFT_SYS + feedback},
                {"role": "user", "content": f"Passages:\n{_passages(hits)}\n\nQuestion: {question}"}]
        d = await gw.call(ctx, msgs, "draft", max_tokens=500, is_draft=True)
        if d.status == "budget_exhausted":
            return await extractive_fallback((hits, top, cov))
        if d.status != "ok":
            return done(_fail_msg(ctx), _fail_status(d), d.error or "draft failed")
        if d.text.strip().startswith(REFUSAL[:30]):
            return done(REFUSAL, "ok", "refused_by_model")
        ok = _cites_valid(d.text, hits)
        if ok and used_mode == "full":
            c = await gw.call(ctx, [{"role": "system", "content": CHECK_SYS},
                                    {"role": "user", "content": f"Passages:\n{_passages(hits)}\n\nAnswer:\n{d.text}"}],
                              "check", max_tokens=120)
            if c.status != "ok":
                # can't verify: the citations passed the code check, so send it but label it
                return done(d.text + "\n\n(Note: the automatic fact-check could not run; citations were verified.)",
                            "ok", "answered_unverified", _norm_cites(d.text))
            ok = c.text.strip().upper().startswith("PASS")
        if ok:
            return done(d.text, "ok", "answered", _norm_cites(d.text))
        feedback = " Your previous answer had unsupported claims or invalid citations; fix that."
    return done(REFUSAL, "ok", "refused_failed_check")
