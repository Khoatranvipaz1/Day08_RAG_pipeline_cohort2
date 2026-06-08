"""Task 5: semantic search module.

Lightweight implementation: TF-IDF cosine similarity stands in for dense
embeddings when local transformer models are not installed.
"""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

from .task4_chunking_indexing import build_vector_index, embed_texts_openai


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    if not query.strip():
        return []
    if os.getenv("USE_WEAVIATE", "").lower() == "true":
        cloud_results = weaviate_search(query, top_k=top_k)
        if cloud_results:
            return cloud_results

    index = build_vector_index()
    chunks = index["chunks"]
    if not chunks or index["matrix"] is None:
        return []

    query_vec = index["vectorizer"].transform([query])
    scores = cosine_similarity(query_vec, index["matrix"]).ravel()
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    results = []
    for idx, score in ranked:
        chunk = chunks[idx]
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": dict(chunk.get("metadata", {})) | {"retriever": "tfidf_semantic"},
            }
        )
    return results


def weaviate_search(query: str, top_k: int = 10, class_name: str = "Day08Chunk") -> list[dict]:
    load_dotenv()
    url = os.getenv("WEAVIATE_URL", "").strip().rstrip("/")
    api_key = os.getenv("WEAVIATE_API_KEY", "").strip()
    if not url:
        return []

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        query_vector = embed_texts_openai([query])[0]
        graphql = {
            "query": f"""
            {{
              Get {{
                {class_name}(nearVector: {{vector: {query_vector}}}, limit: {int(top_k)}) {{
                  content
                  source
                  title
                  path
                  doc_type
                  chunk_id
                  _additional {{ certainty distance }}
                }}
              }}
            }}
            """
        }
        response = requests.post(f"{url}/v1/graphql", headers=headers, json=graphql, timeout=30)
        if response.status_code != 200:
            return []
        objects = response.json().get("data", {}).get("Get", {}).get(class_name, [])
        results = []
        for obj in objects:
            additional = obj.get("_additional", {})
            score = additional.get("certainty")
            if score is None:
                distance = float(additional.get("distance", 1.0))
                score = max(0.0, 1.0 - distance)
            results.append(
                {
                    "content": obj.get("content", ""),
                    "score": float(score),
                    "metadata": {
                        "source": obj.get("source", ""),
                        "title": obj.get("title", ""),
                        "path": obj.get("path", ""),
                        "doc_type": obj.get("doc_type", ""),
                        "chunk_id": obj.get("chunk_id", ""),
                        "retriever": "weaviate_openai",
                    },
                }
            )
        return sorted(results, key=lambda item: item["score"], reverse=True)
    except Exception:
        return []


def main(query: str = "Luật phòng chống ma túy quy định gì?", top_k: int = 5) -> list[dict]:
    return semantic_search(query, top_k=top_k)


if __name__ == "__main__":
    for item in main():
        print(item["score"], item["metadata"].get("source"))
