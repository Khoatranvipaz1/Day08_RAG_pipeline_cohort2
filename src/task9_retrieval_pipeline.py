"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.
"""

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # dùng RRF, không gọi Jina nữa


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        semantic_search + lexical_search
        -> merge bằng RRF
        -> nếu score thấp thì fallback PageIndex
    """
    if not query or not query.strip():
        return []

    # Step 1: Semantic + Lexical search
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    # Step 2: Merge + rerank bằng RRF
    merged = rerank_rrf(
        [dense_results, sparse_results],
        top_k=top_k * 2
    )

    for item in merged:
        item["source"] = "hybrid"

    # Step 3: Vì đã dùng RRF ở trên nên không gọi Jina nữa
    if use_reranking and merged:
        if RERANK_METHOD == "rrf":
            final_results = merged[:top_k]
        else:
            final_results = rerank(
                query=query,
                candidates=merged,
                top_k=top_k,
                method=RERANK_METHOD
            )
    else:
        final_results = merged[:top_k]

    for item in final_results:
        item["source"] = "hybrid"

    # Step 4: Fallback sang PageIndex nếu kết quả yếu
    best_score = final_results[0]["score"] if final_results else 0.0

    if not final_results or best_score < score_threshold:
        print(
            f"  ⚠ Hybrid score ({best_score:.3f}) "
            f"< threshold ({score_threshold}). Fallback → PageIndex"
        )

        try:
            fallback = pageindex_search(query, top_k=top_k)
            return fallback
        except Exception as e:
            print(f"  ⚠ PageIndex fallback failed: {e}")
            return final_results[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)

        results = retrieve(q, top_k=3)

        for i, r in enumerate(results, 1):
            print(
                f"  {i}. [{r['score']:.3f}] "
                f"[{r['source']}] {r['content'][:80]}..."
            )