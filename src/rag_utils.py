"""Shared lightweight helpers for the Day 08 RAG lab.

The project README names heavier production tools such as Crawl4AI,
Weaviate, PageIndex and DeepEval.  This helper keeps the same learning
surface while using local files, TF-IDF and BM25 so the lab runs reliably
on a fresh Windows environment.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LANDING_DIR = DATA_DIR / "landing"
STANDARDIZED_DIR = DATA_DIR / "standardized"
INDEX_DIR = DATA_DIR / "index"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


LEGAL_DOCS = [
    {
        "filename": "luat-phong-chong-ma-tuy-2021.pdf",
        "title": "Luật Phòng, chống ma túy 2021",
        "source": "QuocHoi",
        "year": "2021",
        "content": (
            "Luật Phòng, chống ma túy 2021 quy định trách nhiệm phòng ngừa, "
            "đấu tranh với tội phạm ma túy, quản lý người sử dụng trái phép "
            "chất ma túy và kiểm soát các hoạt động hợp pháp liên quan đến ma túy. "
            "Cơ quan, tổ chức, cá nhân phải phối hợp phát hiện, ngăn chặn và xử lý "
            "hành vi liên quan đến chất ma túy theo quy định pháp luật."
        ),
    },
    {
        "filename": "nghi-dinh-105-2021.pdf",
        "title": "Nghị định 105/2021/NĐ-CP",
        "source": "ChinhPhu",
        "year": "2021",
        "content": (
            "Nghị định 105/2021/NĐ-CP hướng dẫn thi hành Luật Phòng, chống ma túy. "
            "Văn bản quy định chi tiết về quản lý người sử dụng trái phép chất ma túy, "
            "xét nghiệm chất ma túy, hồ sơ quản lý và trách nhiệm phối hợp giữa các "
            "cơ quan trong công tác phòng, chống ma túy."
        ),
    },
    {
        "filename": "bo-luat-hinh-su-2015-chuong-ma-tuy.pdf",
        "title": "Bộ luật Hình sự 2015 - Các tội phạm về ma túy",
        "source": "QuocHoi",
        "year": "2015",
        "content": (
            "Bộ luật Hình sự 2015, sửa đổi 2017, quy định các tội phạm về ma túy như "
            "tàng trữ, vận chuyển, mua bán trái phép chất ma túy. Mức hình phạt phụ "
            "thuộc vào loại chất ma túy, khối lượng, hành vi phạm tội và các tình tiết "
            "tăng nặng hoặc giảm nhẹ."
        ),
    },
    {
        "filename": "tai-lieu-mau-danh-muc-chat-ma-tuy.pdf",
        "title": "Tài liệu mẫu về danh mục chất ma túy",
        "source": "LabDataset",
        "year": "2024",
        "content": (
            "Tài liệu mẫu phục vụ lab RAG mô tả cách quản lý danh mục chất ma túy, "
            "tiền chất và các chất hướng thần. Nội dung nhấn mạnh rằng câu trả lời "
            "cần phân biệt dữ liệu mẫu với văn bản pháp luật chính thức. Khi người "
            "dùng hỏi về danh mục chất, hệ thống cần retrieve đúng chunk có từ khóa "
            "danh mục, tiền chất, kiểm soát đặc biệt và không suy đoán ngoài context. "
            "Metadata source được đặt là LabDataset để minh bạch trong citation."
        ),
    },
    {
        "filename": "tai-lieu-mau-quy-trinh-cai-nghien.pdf",
        "title": "Tài liệu mẫu về quy trình cai nghiện và hỗ trợ cộng đồng",
        "source": "LabDataset",
        "year": "2024",
        "content": (
            "Tài liệu mẫu phục vụ lab RAG mô tả các bước hỗ trợ người sử dụng trái "
            "phép chất ma túy: tiếp nhận thông tin, đánh giá nhu cầu hỗ trợ, tư vấn, "
            "kết nối dịch vụ y tế, theo dõi và tái hòa nhập cộng đồng. Nội dung dùng "
            "để kiểm tra câu hỏi về hỗ trợ, tư vấn, cai nghiện tự nguyện và vai trò "
            "của gia đình, nhà trường, địa phương trong phòng chống ma túy."
        ),
    },
]


NEWS_DOCS = [
    {
        "filename": "news-nghe-si-a.json",
        "title": "Nghệ sĩ A bị điều tra vì liên quan ma túy",
        "source": "BaoDemo",
        "year": "2024",
        "url": "https://example.com/news-a",
        "content": (
            "Một nghệ sĩ Việt Nam bị cơ quan chức năng điều tra sau khi có dấu hiệu "
            "liên quan đến ma túy. Bài viết nhấn mạnh thông tin cần được kiểm chứng "
            "từ nguồn chính thức và không suy đoán ngoài dữ kiện đã công bố."
        ),
    },
    {
        "filename": "news-nghe-si-b.json",
        "title": "Cảnh báo tác hại ma túy trong giới biểu diễn",
        "source": "BaoDemo",
        "year": "2023",
        "url": "https://example.com/news-b",
        "content": (
            "Các chuyên gia cảnh báo việc sử dụng chất ma túy gây ảnh hưởng nghiêm "
            "trọng đến sức khỏe, hình ảnh cá nhân và hoạt động nghệ thuật. Công tác "
            "truyền thông cần dựa trên thông tin chính xác."
        ),
    },
    {
        "filename": "news-nghe-si-c.json",
        "title": "Vụ việc ma túy và trách nhiệm truyền thông",
        "source": "BaoDemo",
        "year": "2022",
        "url": "https://example.com/news-c",
        "content": (
            "Một vụ việc liên quan đến ma túy trong lĩnh vực giải trí cho thấy báo chí "
            "cần trích dẫn nguồn rõ ràng, tránh lan truyền thông tin chưa được xác nhận."
        ),
    },
    {
        "filename": "news-nghe-si-d.json",
        "title": "Cơ quan chức năng xử lý hành vi sử dụng ma túy",
        "source": "BaoDemo",
        "year": "2021",
        "url": "https://example.com/news-d",
        "content": (
            "Theo cơ quan chức năng, hành vi sử dụng trái phép chất ma túy có thể bị "
            "xử lý hành chính hoặc áp dụng biện pháp quản lý tùy trường hợp cụ thể."
        ),
    },
    {
        "filename": "news-nghe-si-e.json",
        "title": "Giáo dục phòng chống ma túy cho người trẻ",
        "source": "BaoDemo",
        "year": "2024",
        "url": "https://example.com/news-e",
        "content": (
            "Các chương trình giáo dục phòng chống ma túy tập trung vào nhận biết rủi ro, "
            "kỹ năng từ chối và tìm kiếm hỗ trợ từ gia đình, nhà trường, cơ quan chức năng."
        ),
    },
    {
        "filename": "news-cong-dong-f.json",
        "title": "Chiến dịch truyền thông phòng chống ma túy tại cộng đồng",
        "source": "BaoDemo",
        "year": "2024",
        "url": "https://example.com/news-f",
        "content": (
            "Một chiến dịch truyền thông tại cộng đồng tập trung vào nhận diện dấu hiệu "
            "rủi ro, khuyến khích người trẻ tìm kiếm hỗ trợ sớm và tăng phối hợp giữa "
            "gia đình, nhà trường, địa phương. Bài viết mẫu dùng để kiểm tra retrieval "
            "cho nhóm câu hỏi về phòng ngừa và truyền thông."
        ),
    },
    {
        "filename": "news-chinh-sach-g.json",
        "title": "Chuyên gia khuyến nghị không suy đoán khi đưa tin về ma túy",
        "source": "BaoDemo",
        "year": "2025",
        "url": "https://example.com/news-g",
        "content": (
            "Chuyên gia truyền thông khuyến nghị các nền tảng tin tức chỉ công bố thông tin "
            "đã được xác minh, tránh nêu danh tính hoặc dữ liệu cá nhân khi không có căn cứ. "
            "Nội dung này hỗ trợ kiểm tra grounding, abstention và bảo vệ thông tin nhạy cảm."
        ),
    },
    {
        "filename": "news-y-te-h.json",
        "title": "Hỗ trợ y tế và tâm lý cho người cần cai nghiện",
        "source": "BaoDemo",
        "year": "2024",
        "url": "https://example.com/news-h",
        "content": (
            "Các cơ sở hỗ trợ y tế và tâm lý có thể giúp người cần cai nghiện đánh giá rủi ro, "
            "lập kế hoạch điều trị, theo dõi sức khỏe và kết nối với dịch vụ xã hội. Bài viết "
            "mẫu dùng cho câu hỏi về hỗ trợ cộng đồng, cai nghiện tự nguyện và tái hòa nhập."
        ),
    },
]


def ensure_dirs() -> None:
    for directory in [
        LANDING_DIR / "legal",
        LANDING_DIR / "news",
        STANDARDIZED_DIR / "legal",
        STANDARDIZED_DIR / "news",
        INDEX_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def write_sample_landing_data(force: bool = False) -> None:
    """Create small local sample files when the user has not collected data yet."""
    ensure_dirs()
    for item in LEGAL_DOCS:
        path = LANDING_DIR / "legal" / item["filename"]
        if force or not path.exists():
            path.write_text(item["content"], encoding="utf-8")

    for item in NEWS_DOCS:
        path = LANDING_DIR / "news" / item["filename"]
        if force or not path.exists():
            path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")


def standardize_landing_data(force: bool = False) -> list[Path]:
    """Convert local landing files into Markdown documents."""
    write_sample_landing_data()
    outputs: list[Path] = []

    for item in LEGAL_DOCS:
        src = LANDING_DIR / "legal" / item["filename"]
        dst = STANDARDIZED_DIR / "legal" / (Path(item["filename"]).stem + ".md")
        if force or not dst.exists():
            text = src.read_text(encoding="utf-8", errors="replace")
            dst.write_text(
                f"# {item['title']}\n\n"
                f"- Source: {item['source']}\n"
                f"- Year: {item['year']}\n"
                f"- Type: legal\n\n"
                f"{text}\n",
                encoding="utf-8",
            )
        outputs.append(dst)

    for item in NEWS_DOCS:
        src = LANDING_DIR / "news" / item["filename"]
        dst = STANDARDIZED_DIR / "news" / (Path(item["filename"]).stem + ".md")
        if force or not dst.exists():
            data = json.loads(src.read_text(encoding="utf-8"))
            dst.write_text(
                f"# {data['title']}\n\n"
                f"- Source: {data['source']}\n"
                f"- Year: {data['year']}\n"
                f"- URL: {data['url']}\n"
                f"- Type: news\n\n"
                f"{data['content']}\n",
                encoding="utf-8",
            )
        outputs.append(dst)

    return outputs


def tokenize(text: str) -> list[str]:
    text = strip_accents(text.lower())
    return re.findall(r"[\w]+", text, flags=re.UNICODE)


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def load_markdown_documents() -> list[dict]:
    standardize_landing_data()
    docs = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        doc_type = "news" if "news" in path.parts else "legal"
        title = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
        docs.append(
            {
                "content": text,
                "metadata": {
                    "source": path.stem,
                    "path": str(path.relative_to(ROOT)),
                    "doc_type": doc_type,
                    "title": title,
                },
            }
        )
    return docs


def split_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks(chunk_size: int = 900, overlap: int = 120, persist: bool = True) -> list[dict]:
    docs = load_markdown_documents()
    chunks: list[dict] = []
    for doc in docs:
        parts = split_text(doc["content"], chunk_size=chunk_size, overlap=overlap)
        for idx, content in enumerate(parts):
            metadata = dict(doc["metadata"])
            metadata["chunk_id"] = f"{metadata['source']}::{idx}"
            chunks.append({"content": content, "score": 0.0, "metadata": metadata})

    if persist:
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return chunks


def get_chunks() -> list[dict]:
    if CHUNKS_PATH.exists():
        try:
            chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
            if chunks:
                return chunks
        except json.JSONDecodeError:
            pass
    return build_chunks()


def normalize_scores(items: Iterable[dict]) -> list[dict]:
    results = [dict(item) for item in items]
    if not results:
        return []
    scores = [float(item.get("score", 0.0)) for item in results]
    low, high = min(scores), max(scores)
    for item in results:
        score = float(item.get("score", 0.0))
        item["score"] = 1.0 if math.isclose(high, low) and high > 0 else (
            0.0 if math.isclose(high, low) else (score - low) / (high - low)
        )
    return results


def citation_for(chunk: dict, fallback_index: int = 1) -> str:
    metadata = chunk.get("metadata", {})
    source = metadata.get("source") or metadata.get("title") or f"doc_{fallback_index}"
    return f"[{source}]"
