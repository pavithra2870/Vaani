"""Split the three PDFs into chunks. Each chunk keeps paper + page so answers can cite them.

Chunks never cross a page boundary (simple, and makes page citations exact).
Trade-off: a sentence split across pages is cut in two.

Run:  python -m relay.ingest
"""
import json
import re
import fitz  # PyMuPDF
from .config import DATA_DIR, INDEX_DIR, PAPERS, RETRIEVAL_SETTINGS


def clean(text: str) -> str:
    text = text.replace("­", "")
    text = re.sub(r"-\n(\w)", r"\1", text)      # re-join hyphenated line breaks
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_pages(pdf_path):
    doc = fitz.open(pdf_path)
    # Use the page number printed in the PDF order (1-based); this is the PDF page index.
    return [(i + 1, clean(p.get_text())) for i, p in enumerate(doc)]


def chunk_page(words, size, overlap):
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) < 20 and start > 0:  # drop tiny tail
            break
        yield " ".join(piece)
        if start + size >= len(words):
            break


def build(setting_name: str):
    s = RETRIEVAL_SETTINGS[setting_name]
    chunks = []
    for key, meta in PAPERS.items():
        for page, text in read_pages(DATA_DIR / meta["file"]):
            words = text.split()
            for i, piece in enumerate(chunk_page(words, s["chunk_words"], s["overlap"])):
                chunks.append({
                    "id": f"{key}-p{page}-c{i}",
                    "paper": key,
                    "arxiv": meta["arxiv"],
                    "title": meta["title"],
                    "page": page,
                    "text": piece,
                })
    return chunks


def main():
    INDEX_DIR.mkdir(exist_ok=True)
    # settings that share chunking params share an index file
    done = {}
    for name, s in RETRIEVAL_SETTINGS.items():
        key = (s["chunk_words"], s["overlap"])
        if key in done:
            continue
        chunks = build(name)
        path = INDEX_DIR / f"chunks_{s['chunk_words']}_{s['overlap']}.json"
        path.write_text(json.dumps(chunks), encoding="utf-8")
        done[key] = path
        print(f"{name}: {len(chunks)} chunks -> {path.name}")


if __name__ == "__main__":
    main()
