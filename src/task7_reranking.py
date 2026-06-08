"""Task 7: fusion and reranking."""

from __future__ import annotations

from collections import defaultdict
import os

import requests
from dotenv import load_dotenv

from .rag_utils import tokenize


def _doc_key(item: dict) -> str:
    metadata = item.get("metadata", {})
    return metadata.get("chunk_id") or metadata.get("path") or item.get("content", "")[:120]


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60, top_k: int = 10) -> list[dict]:
    fused_scores: dict[str, float] = defaultdict(float)
    docs: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            key = _doc_key(item)
            fused_scores[key] += 1.0 / (k + rank)
            docs.setdefault(key, item)

    fused = []
    for key, item in docs.items():
        merged = {
            "content": item["content"],
            "score": float(fused_scores[key]),
            "metadata": dict(item.get("metadata", {})) | {"fusion": "rrf"},
        }
        fused.append(merged)
    return sorted(fused, key=lambda item: item["score"], reverse=True)[:top_k]


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    jina_results = _jina_rerank(query, candidates, top_k=top_k)
    if jina_results is not None:
        return jina_results

    query_terms = set(tokenize(query))
    if not query_terms:
        return candidates[:top_k]

    reranked = []
    for item in candidates:
        doc_terms = set(tokenize(item.get("content", "")))
        overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
        base_score = float(item.get("score", 0.0))
        score = 0.7 * base_score + 0.3 * overlap
        reranked.append(
            {
                "content": item["content"],
                "score": float(score),
                "metadata": dict(item.get("metadata", {})) | {"reranker": "term_overlap"},
            }
        )
    return sorted(reranked, key=lambda item: item["score"], reverse=True)[:top_k]


def _jina_rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict] | None:
    """Use Jina reranker when JINA_API_KEY is present.

    Falls back silently to local term-overlap reranking on any API/network issue.
    """
    load_dotenv()
    api_key = os.getenv("JINA_API_KEY", "").strip()
    model = os.getenv("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual").strip()
    if not api_key or not candidates:
        return None

    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "query": query,
                "documents": [item.get("content", "") for item in candidates],
                "top_n": min(top_k, len(candidates)),
            },
            timeout=30,
        )
        if response.status_code != 200:
            return None
        payload = response.json()
        reranked = []
        for result in payload.get("results", []):
            index = int(result["index"])
            source = candidates[index]
            reranked.append(
                {
                    "content": source["content"],
                    "score": float(result.get("relevance_score", source.get("score", 0.0))),
                    "metadata": dict(source.get("metadata", {})) | {"reranker": "jina", "jina_model": model},
                }
            )
        return reranked[:top_k]
    except Exception:
        return None


def main(query: str = "phòng chống ma túy", top_k: int = 5) -> list[dict]:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search

    fused = reciprocal_rank_fusion([semantic_search(query), lexical_search(query)], top_k=top_k * 2)
    return rerank(query, fused, top_k=top_k)


if __name__ == "__main__":
    for item in main():
        print(item["score"], item["metadata"].get("source"))
