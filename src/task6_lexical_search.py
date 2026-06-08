"""Task 6: lexical BM25 search."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from .rag_utils import get_chunks, tokenize


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    if not query.strip():
        return []
    chunks = get_chunks()
    if not chunks:
        return []

    tokenized_corpus = [tokenize(chunk["content"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]

    results = []
    for idx, score in ranked:
        chunk = chunks[idx]
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": dict(chunk.get("metadata", {})) | {"retriever": "bm25"},
            }
        )
    return results


def main(query: str = "Nghị định 105 ma túy", top_k: int = 5) -> list[dict]:
    return lexical_search(query, top_k=top_k)


if __name__ == "__main__":
    for item in main():
        print(item["score"], item["metadata"].get("source"))
