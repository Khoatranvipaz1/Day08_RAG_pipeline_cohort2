"""Task 8: PageIndex/vectorless fallback.

This is a local fallback implementation.  It keeps the same public function
as the PageIndex requirement but searches local chunks without external API.
"""

from __future__ import annotations

from .rag_utils import get_chunks, tokenize


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    query_terms = set(tokenize(query))
    chunks = get_chunks()
    if not query_terms or not chunks:
        return []

    results = []
    for chunk in chunks:
        doc_terms = set(tokenize(chunk["content"]))
        overlap = len(query_terms & doc_terms)
        if overlap == 0:
            continue
        score = overlap / max(len(query_terms), 1)
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": dict(chunk.get("metadata", {})) | {"retriever": "local_vectorless_fallback"},
            }
        )
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def main(query: str = "ma túy", top_k: int = 5) -> list[dict]:
    return pageindex_search(query, top_k=top_k)


if __name__ == "__main__":
    for item in main():
        print(item["score"], item["metadata"].get("source"))
