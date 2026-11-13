"""BM25 retrieval over the chunk index, with an optional cheap re-rank. No model calls, no quota used."""
import json
import math
import re
from collections import Counter
from functools import lru_cache
from .config import INDEX_DIR, RETRIEVAL_SETTINGS

STOP = set("the a an of to in and or is are was were be for on with as by that this it its from at which what how why does do did can we our their".split())


def tok(text: str):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tok(c["text"])) for c in chunks]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / max(1, len(self.len))
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query_tokens):
        out = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in query_tokens:
                f = d.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                s += self.idf[t] * f * (self.k1 + 1) / denom
            out.append(s)
        return out


@lru_cache(maxsize=None)
def _index(chunk_words: int, overlap: int):
    path = INDEX_DIR / f"chunks_{chunk_words}_{overlap}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run: python -m relay.ingest")
    return BM25(json.loads(path.read_text(encoding="utf-8")))


def _rerank(query_tokens, hits):
    """Cheap re-rank: favour chunks covering more DISTINCT query terms (BM25 alone rewards repetition)."""
    qs = set(query_tokens)
    def cov(h):
        return len(qs & set(tok(h["text"]))) / max(1, len(qs))
    return sorted(hits, key=lambda h: (h["score"] * (0.5 + cov(h))), reverse=True)


def retrieve(query: str, setting: str, pool: int = 20):
    """Returns (hits, top_score, coverage). Each hit = chunk dict + 'score'.
    coverage = idf-weighted share of the query's terms that occur somewhere in the retrieved passages."""
    s = RETRIEVAL_SETTINGS[setting]
    idx = _index(s["chunk_words"], s["overlap"])
    qt = tok(query)
    scores = idx.score(qt)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    n = pool if s["rerank"] else s["top_k"]
    hits = [dict(idx.chunks[i], score=scores[i]) for i in order[:n] if scores[i] > 0]
    if s["rerank"]:
        hits = _rerank(qt, hits)[: s["top_k"]]
    present = set()
    for h in hits:
        present |= set(tok(h["text"]))
    qs = set(qt)
    w = lambda terms: sum(idx.idf.get(x, max(idx.idf.values())) for x in terms)
    cov = w(qs & present) / w(qs) if qs else 0.0
    return hits, (hits[0]["score"] if hits else 0.0), cov
