"""Task 4: chunking and lightweight indexing."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

import requests
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

from .rag_utils import CHUNKS_PATH, build_chunks, get_chunks, load_markdown_documents, split_text

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
EMBEDDING_MODEL = "sklearn:TfidfVectorizer"
EMBEDDING_DIMENSION = "vocabulary_size"


def load_documents() -> list[dict]:
    return load_markdown_documents()


def chunk_documents(
    documents: list[dict] | None = None,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    if documents is None:
        return build_chunks(chunk_size=chunk_size, overlap=overlap, persist=True)

    chunks: list[dict] = []
    for doc in documents:
        metadata = dict(doc.get("metadata", {}))
        source = metadata.get("source", "doc")
        for idx, content in enumerate(split_text(doc.get("content", ""), chunk_size, overlap)):
            item_metadata = dict(metadata)
            item_metadata["chunk_id"] = f"{source}::{idx}"
            chunks.append({"content": content, "score": 0.0, "metadata": item_metadata})
    return chunks


def build_vector_index(chunks: list[dict] | None = None) -> dict:
    chunks = chunks or get_chunks()
    texts = [chunk["content"] for chunk in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    matrix = vectorizer.fit_transform(texts) if texts else None
    return {
        "chunks": chunks,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "metadata": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dimension": len(vectorizer.vocabulary_) if texts else 0,
            "index_path": str(CHUNKS_PATH),
        },
    }


def embed_texts_openai(texts: list[str]) -> list[list[float]]:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
        try:
            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]
        except Exception:
            pass
    return embed_texts_jina(texts)


def embed_texts_jina(texts: list[str]) -> list[list[float]]:
    load_dotenv()
    api_key = os.getenv("JINA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("JINA_API_KEY is required when OpenAI embeddings are unavailable")
    model = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3").strip()
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "input": texts},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    return [item["embedding"] for item in payload.get("data", [])]


def index_to_weaviate(chunks: list[dict] | None = None, class_name: str = "Day08Chunk") -> dict:
    """Index chunks into Weaviate Cloud/local using REST + OpenAI embeddings.

    This optional path is the full-stack replacement for the local TF-IDF index.
    It is not called by default so offline tests remain fast and reliable.
    """
    load_dotenv()
    url = os.getenv("WEAVIATE_URL", "").strip().rstrip("/")
    api_key = os.getenv("WEAVIATE_API_KEY", "").strip()
    if not url:
        raise RuntimeError("WEAVIATE_URL is required")

    chunks = chunks or get_chunks()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    schema = {
        "class": class_name,
        "description": "Day 08 RAG lab chunks",
        "vectorizer": "none",
        "properties": [
            {"name": "content", "dataType": ["text"]},
            {"name": "source", "dataType": ["text"]},
            {"name": "title", "dataType": ["text"]},
            {"name": "path", "dataType": ["text"]},
            {"name": "doc_type", "dataType": ["text"]},
            {"name": "chunk_id", "dataType": ["text"]},
        ],
    }
    existing = requests.get(f"{url}/v1/schema/{class_name}", headers=headers, timeout=20)
    if existing.status_code == 404:
        created = requests.post(f"{url}/v1/schema", headers=headers, json=schema, timeout=30)
        created.raise_for_status()
    elif existing.status_code not in {200, 204}:
        existing.raise_for_status()

    vectors = embed_texts_openai([chunk["content"] for chunk in chunks])
    imported = 0
    for chunk, vector in zip(chunks, vectors):
        metadata = chunk.get("metadata", {})
        chunk_id = metadata.get("chunk_id") or metadata.get("path") or chunk["content"][:80]
        obj = {
            "class": class_name,
            "id": str(uuid5(NAMESPACE_URL, chunk_id)),
            "properties": {
                "content": chunk["content"],
                "source": metadata.get("source", ""),
                "title": metadata.get("title", ""),
                "path": metadata.get("path", ""),
                "doc_type": metadata.get("doc_type", ""),
                "chunk_id": chunk_id,
            },
            "vector": vector,
        }
        response = requests.post(f"{url}/v1/objects", headers=headers, json=obj, timeout=30)
        if response.status_code in {409, 422}:
            response = requests.put(
                f"{url}/v1/objects/{class_name}/{obj['id']}",
                headers=headers,
                json=obj,
                timeout=30,
            )
        response.raise_for_status()
        imported += 1

    return {"class_name": class_name, "imported": imported, "weaviate_url": url}


def chunking_indexing() -> dict:
    chunks = chunk_documents()
    index = build_vector_index(chunks)
    result = {
        "num_documents": len(load_documents()),
        "num_chunks": len(chunks),
        "index_path": str(Path(CHUNKS_PATH)),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": index["metadata"]["embedding_dimension"],
    }
    if os.getenv("USE_WEAVIATE", "").lower() == "true":
        result["weaviate"] = index_to_weaviate(chunks)
    return result


def main() -> dict:
    return chunking_indexing()


if __name__ == "__main__":
    print(main())
