"""Task 3: convert collected data to Markdown."""

from __future__ import annotations

import json
from pathlib import Path

from .rag_utils import LANDING_DIR, STANDARDIZED_DIR, standardize_landing_data


def convert_file_to_markdown(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    src = Path(input_path)
    doc_type = "news" if "news" in src.parts else "legal"
    dst = Path(output_path) if output_path else STANDARDIZED_DIR / doc_type / f"{src.stem}.md"
    dst.parent.mkdir(parents=True, exist_ok=True)

    raw = src.read_text(encoding="utf-8", errors="replace")
    if src.suffix.lower() == ".json":
        data = json.loads(raw)
        markdown = (
            f"# {data.get('title', src.stem)}\n\n"
            f"- Source: {data.get('source', 'unknown')}\n"
            f"- URL: {data.get('url', '')}\n"
            f"- Crawl date: {data.get('crawl_date', '')}\n"
            f"- Type: news\n\n"
            f"{data.get('content', '')}\n"
        )
    else:
        markdown = f"# {src.stem}\n\n- Source: {src.name}\n- Type: {doc_type}\n\n{raw}\n"
    dst.write_text(markdown, encoding="utf-8")
    return dst


def convert_all_to_markdown(
    input_dir: str | Path = LANDING_DIR,
    output_dir: str | Path = STANDARDIZED_DIR,
) -> list[Path]:
    # Ensure default sample data exists, then convert any current landing files.
    standardize_landing_data()
    input_base = Path(input_dir)
    output_base = Path(output_dir)
    outputs: list[Path] = []
    for src in sorted(input_base.rglob("*")):
        if not src.is_file():
            continue
        if src.suffix.lower() not in {".pdf", ".docx", ".html", ".htm", ".json", ".txt", ".md"}:
            continue
        doc_type = "news" if "news" in src.parts else "legal"
        dst = output_base / doc_type / f"{src.stem}.md"
        outputs.append(convert_file_to_markdown(src, dst))
    return outputs


def main() -> list[Path]:
    return convert_all_to_markdown()


if __name__ == "__main__":
    for path in main():
        print(path)
