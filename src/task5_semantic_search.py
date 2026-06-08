"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_DIR = Path(__file__).parent.parent
VECTORSTORE_DIR = PROJECT_DIR / "data" / "vectorstore"

COLLECTION_NAME = "drug_law_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


_model = None


def get_model():
    """Load model 1 lần, tránh tải lại nhiều lần."""
    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)

    return _model


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

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

    # Bước 1: Embed query bằng cùng model ở Task 4
    model = get_model()
    query_embedding = model.encode(query).tolist()

    # Bước 2: Kết nối ChromaDB đã tạo ở Task 4
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    # Bước 3: Query vector store
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    output = []

    for content, metadata, distance in zip(documents, metadatas, distances):
        # ChromaDB trả distance, distance càng nhỏ càng giống
        # Đổi sang score: score càng lớn càng tốt
        score = 1 / (1 + distance)

        output.append({
            "content": content,
            "score": float(score),
            "metadata": metadata or {}
        })

    # Sort score giảm dần
    output = sorted(output, key=lambda x: x["score"], reverse=True)

    return output[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)

    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
        print("Metadata:", r["metadata"])
        print()