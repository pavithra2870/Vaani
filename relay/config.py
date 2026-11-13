"""Central config. Everything tunable lives here or in .env."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"
RUNS_DIR = ROOT / "runs"
CACHE_DIR = ROOT / ".cache"

BACKEND = os.getenv("LLM_BACKEND", "standin")  # standin | groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
STANDIN_URL = os.getenv("STANDIN_URL", "http://127.0.0.1:8001") + "/v1/chat/completions"
GROQ_DAILY_TOKEN_CAP = int(os.getenv("GROQ_DAILY_TOKEN_CAP", "150000"))
LLM_CACHE = os.getenv("LLM_CACHE", "0") == "1"

# The library: exactly these versions (assignment requirement).
PAPERS = {
    "attention": {"file": "1706.03762v7.pdf", "title": "Attention Is All You Need", "arxiv": "1706.03762v7"},
    "moshi": {"file": "2410.00037v1.pdf", "title": "Moshi", "arxiv": "2410.00037v1"},
    "dynchunk": {"file": "2507.07955v2.pdf", "title": "Dynamic Chunking", "arxiv": "2507.07955v2"},
}

# Retrieval settings to compare (Table 3). chunk_words, top_k, rerank
RETRIEVAL_SETTINGS = {
    "small-k8":        {"chunk_words": 120, "overlap": 30, "top_k": 8, "rerank": False},
    "medium-k4":       {"chunk_words": 250, "overlap": 50, "top_k": 4, "rerank": False},
    "medium-k4-rerank": {"chunk_words": 250, "overlap": 50, "top_k": 4, "rerank": True},
    "large-k2":        {"chunk_words": 500, "overlap": 80, "top_k": 2, "rerank": False},
}
DEFAULT_SETTING = "small-k8"

# Retrieval "decide" branch: refuse WITHOUT a draft call only when the evidence is clearly off-topic.
# Calibrated offline (eval/offline_retrieval.py): lexical scores cannot separate on-topic-but-unanswered questions
# (those must be refused by the model), so the gate is set just below the weakest answerable question
# (BM25 top 6.2, idf-coverage 0.59) and only catches questions that are clearly about something else.
RELEVANCE_MIN = float(os.getenv("RELEVANCE_MIN", "5.0"))
COVERAGE_MIN = float(os.getenv("COVERAGE_MIN", "0.5"))
