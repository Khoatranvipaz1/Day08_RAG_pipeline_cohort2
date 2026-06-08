"""
Task 7 — Reranking Module.

Sử dụng Jina Reranker v2 multilingual làm cross-encoder reranker.

Lý do chọn:
    - Hỗ trợ multilingual, phù hợp tiếng Việt
    - Không cần chạy model nặng local
    - Rerank dựa trên cả query và document, tốt hơn chỉ fusion điểm
"""

import os
from dotenv import load_dotenv


load_dotenv()


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng Jina Reranker API.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by score descending.
    """
    import requests

    if not query or not query.strip():
        return []

    if not candidates:
        return []

    api_key = os.getenv("JINA_API_KEY")

    if not api_key:
        raise ValueError("Missing JINA_API_KEY. Please add JINA_API_KEY to .env")

    documents = [c["content"] for c in candidates]

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),
        },
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()
    reranked = data.get("results", [])

    results = []

    for r in reranked:
        original_index = r["index"]
        relevance_score = r["relevance_score"]

        item = candidates[original_index].copy()
        item["score"] = float(relevance_score)
        item["rerank_score"] = float(relevance_score)
        item["rerank_method"] = "jina_cross_encoder"

        results.append(item)

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Không dùng trong bài này.
    Giữ lại để đúng interface của file gốc.
    """
    raise NotImplementedError("This project uses Jina cross-encoder reranking.")


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion fallback nếu cần gộp nhiều ranked lists.

    RRF(d) = Σ 1 / (k + rank_r(d))
    """
    rrf_scores = {}
    content_map = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item.get("content", "")

            if not key:
                continue

            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []

    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        item["rerank_score"] = float(score)
        item["rerank_method"] = "rrf"
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)

    elif method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)

    elif method == "mmr":
        raise NotImplementedError("Call rerank_mmr with query_embedding")

    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {
            "content": "Điều 249 Bộ luật Hình sự quy định về tội tàng trữ trái phép chất ma túy.",
            "score": 0.8,
            "metadata": {"source": "bo_luat_hinh_su.md", "chunk_index": 1},
        },
        {
            "content": "Một nghệ sĩ bị bắt vì sử dụng trái phép chất ma túy.",
            "score": 0.7,
            "metadata": {"source": "news.md", "chunk_index": 2},
        },
        {
            "content": "Thời tiết hôm nay có mưa ở nhiều nơi.",
            "score": 0.6,
            "metadata": {"source": "other.md", "chunk_index": 3},
        },
    ]

    results = rerank(
        "hình phạt tàng trữ trái phép chất ma túy",
        dummy_candidates,
        top_k=2,
        method="cross_encoder",
    )

    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
        print("Metadata:", r["metadata"])
        print("Method:", r["rerank_method"])
        print()