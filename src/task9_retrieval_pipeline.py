"""Task 9: complete retrieval pipeline."""

from __future__ import annotations

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import reciprocal_rank_fusion, rerank
from .task8_pageindex_vectorless import pageindex_search


def retrieve(query: str, top_k: int = 5, score_threshold: float = 0.3) -> list[dict]:
    semantic_results = semantic_search(query, top_k=max(top_k * 2, 10))
    lexical_results = lexical_search(query, top_k=max(top_k * 2, 10))
    fused = reciprocal_rank_fusion([semantic_results, lexical_results], top_k=max(top_k * 3, 15))
    ranked = rerank(query, fused, top_k=top_k)

    if not ranked or float(ranked[0].get("score", 0.0)) < score_threshold:
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return [_with_source(item, "fallback") for item in fallback]

    return [_with_source(item, "hybrid_rrf_rerank") for item in ranked[:top_k]]


def _with_source(item: dict, pipeline: str) -> dict:
    metadata = dict(item.get("metadata", {})) | {"pipeline": pipeline}
    return {
        "content": item["content"],
        "score": item["score"],
        "source": "pageindex" if pipeline == "fallback" else "hybrid",
        "metadata": metadata,
    }


def main(query: str = "Luật phòng chống ma túy quy định trách nhiệm gì?", top_k: int = 5) -> list[dict]:
    return retrieve(query, top_k=top_k)


if __name__ == "__main__":
    for item in main():
        print(item["score"], item["metadata"].get("source"), item["metadata"].get("pipeline"))
