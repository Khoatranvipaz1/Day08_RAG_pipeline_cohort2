"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

"""
Task 8 — PageIndex Vectorless RAG.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
STATE_FILE = Path(__file__).parent.parent / "data" / "pageindex_state.json"


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        raise ValueError("Missing PAGEINDEX_API_KEY in .env")

    uploaded = []

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        print(f"Uploading: {md_file.name}")

        with open(md_file, "rb") as f:
            response = requests.post(
                "https://api.pageindex.ai/markdown/",
                headers={"api_key": PAGEINDEX_API_KEY},
                files={"file": f},
                data={
                    "if_add_node_id": "yes",
                    "if_add_node_text": "yes",
                    "if_add_node_summary": "yes",
                },
                timeout=120,
            )

        if response.status_code != 200:
            print(f"  ✗ Failed: {md_file.name}")
            print(f"  Status: {response.status_code}")
            print(f"  Error: {response.text[:500]}")
            continue

        uploaded.append({
            "filename": md_file.name,
            "type": md_file.parent.name,
            "result": response.json(),
        })

        print(f"  ✓ Uploaded: {md_file.name}")

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(
        json.dumps(uploaded, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return uploaded


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    """
    if not query.strip():
        return []

    if not STATE_FILE.exists():
        upload_documents()

    docs = json.loads(STATE_FILE.read_text(encoding="utf-8"))

    query_words = set(query.lower().split())
    results = []

    for doc in docs:
        structure = doc["result"].get("structure", [])

        stack = structure if isinstance(structure, list) else [structure]

        while stack:
            node = stack.pop()

            if not isinstance(node, dict):
                continue

            title = node.get("title", "")
            text = node.get("text", "")
            summary = node.get("summary", "")
            content = f"{title}\n{summary}\n{text}".strip()

            if content:
                content_words = set(content.lower().split())
                overlap = query_words.intersection(content_words)

                if overlap:
                    score = len(overlap) / max(len(query_words), 1)

                    results.append({
                        "content": content,
                        "score": float(score),
                        "metadata": {
                            "source": doc["filename"],
                            "type": doc["type"],
                            "title": title,
                            "node_id": node.get("node_id", ""),
                        },
                        "source": "pageindex",
                    })

            children = node.get("nodes", [])
            if isinstance(children, list):
                stack.extend(children)

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)

        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")