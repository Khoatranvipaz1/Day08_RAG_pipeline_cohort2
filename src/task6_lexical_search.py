"""
Task 6 — Lexical Search Module (BM25).

BM25 search trên các chunks markdown từ data/standardized/.
"""

from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def load_corpus() -> list[dict]:
    """
    Load markdown files từ data/standardized/ rồi chia chunk đơn giản.
    """
    corpus = []

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"

        # Chia chunk đơn giản theo đoạn văn
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        for i, paragraph in enumerate(paragraphs):
            if len(paragraph) < 30:
                continue

            corpus.append({
                "content": paragraph,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "chunk_index": i
                }
            })

    return corpus


# Load corpus global
CORPUS: list[dict] = load_corpus()


def tokenize(text: str) -> list[str]:
    """
    Tokenize đơn giản cho BM25.
    Dùng lower().split() là đủ cho bài này.
    """
    return text.lower().split()


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25


# Build BM25 index global
BM25_INDEX = build_bm25_index(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict
        }
        Sorted by score descending.
    """
    if not query or not query.strip():
        return []

    tokenized_query = tokenize(query)
    scores = BM25_INDEX.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        score = scores[idx]

        if score > 0:
            results.append({
                "content": CORPUS[idx]["content"],
                "score": float(score),
                "metadata": CORPUS[idx]["metadata"]
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)

    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
        print("Metadata:", r["metadata"])
        print()