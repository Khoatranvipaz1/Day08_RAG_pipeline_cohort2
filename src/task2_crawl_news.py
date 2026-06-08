"""Task 2: crawl news articles.

Uses requests + BeautifulSoup for real URLs when provided.  Without URLs,
it writes five local JSON article files so the rest of the RAG pipeline can
run offline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .rag_utils import LANDING_DIR, NEWS_DOCS, write_sample_landing_data


def _safe_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(part for part in slug.split("-") if part)[:80] or "article"


def crawl_article(url: str, output_dir: str | Path | None = None) -> dict:
    output = Path(output_dir) if output_dir else LANDING_DIR / "news"
    output.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=20, headers={"User-Agent": "Day08RAGLab/1.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html5lib")
    title = soup.title.get_text(" ", strip=True) if soup.title else urlparse(url).netloc
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    content = "\n".join(p for p in paragraphs if p)
    if not content:
        content = soup.get_text(" ", strip=True)[:5000]

    item = {
        "url": url,
        "title": title,
        "source": urlparse(url).netloc,
        "crawl_date": datetime.now(timezone.utc).isoformat(),
        "content": content,
    }
    path = output / f"{_safe_slug(title)}.json"
    path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    return item | {"path": str(path)}


def crawl_news(urls: list[str] | None = None, output_dir: str | Path | None = None) -> list[dict]:
    if urls:
        return [crawl_article(url, output_dir=output_dir) for url in urls]

    write_sample_landing_data()
    base = Path(output_dir) if output_dir else LANDING_DIR / "news"
    base.mkdir(parents=True, exist_ok=True)
    results = []
    for item in NEWS_DOCS:
        path = base / item["filename"]
        data = dict(item)
        data["crawl_date"] = datetime.now(timezone.utc).isoformat()
        if not path.exists():
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(data | {"path": str(path)})
    return results


def main() -> list[dict]:
    return crawl_news()


if __name__ == "__main__":
    for article in main():
        print(article["path"])
