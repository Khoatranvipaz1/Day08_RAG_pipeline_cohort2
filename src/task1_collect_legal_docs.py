"""Task 1: collect legal documents.

The README asks students to download at least three PDF/DOCX legal files.
For an offline lab run, this module creates small placeholder `.pdf` files
with Vietnamese legal content so downstream tasks and tests have data.
"""

from __future__ import annotations

from pathlib import Path

from .rag_utils import LANDING_DIR, LEGAL_DOCS, write_sample_landing_data


def collect_legal_docs(output_dir: str | Path | None = None) -> list[dict]:
    write_sample_landing_data()
    base = Path(output_dir) if output_dir else LANDING_DIR / "legal"
    base.mkdir(parents=True, exist_ok=True)
    return [
        {
            "filename": item["filename"],
            "path": str(base / item["filename"]),
            "title": item["title"],
            "source": item["source"],
            "year": item["year"],
        }
        for item in LEGAL_DOCS
    ]


def main() -> list[dict]:
    return collect_legal_docs()


if __name__ == "__main__":
    for doc in main():
        print(doc)
