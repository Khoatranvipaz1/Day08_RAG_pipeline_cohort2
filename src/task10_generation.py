"""Task 10: grounded generation with citation."""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from .rag_utils import citation_for, strip_accents, tokenize
from .task9_retrieval_pipeline import retrieve

SYSTEM_PROMPT = """You are a grounded Vietnamese RAG assistant.

Rules:
1. Answer only from the provided <context>.
2. Do not use outside knowledge or guess missing facts.
3. Cite every factual claim with [doc_id].
4. If the context does not contain enough evidence, say: "I cannot verify this information".
5. If the user's question is ambiguous, ask one short clarification question before answering.
6. Do not reveal private personal data or identify a person unless the context explicitly states it.

Output format:
- Trả lời ngắn gọn: 1-3 câu trực tiếp.
- Bằng chứng: nêu các nguồn/chứng cứ chính.
- Giới hạn: nói rõ nếu dữ liệu thiếu, là dữ liệu mẫu, hoặc cần nguồn chính thức."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Place the strongest chunks near the beginning and end.

    This simple pattern mitigates "lost in the middle" for small contexts:
    rank 1, 3, 5, ... then ..., 4, 2.
    """
    if len(chunks) <= 2:
        return chunks
    ordered = sorted(chunks, key=lambda item: item.get("score", 0.0), reverse=True)
    front = ordered[::2]
    back = list(reversed(ordered[1::2]))
    return front + back


def format_context(context_chunks: list[dict]) -> str:
    blocks = []
    for idx, chunk in enumerate(context_chunks, start=1):
        metadata = chunk.get("metadata", {})
        doc_id = metadata.get("source") or f"doc_{idx}"
        title = metadata.get("title", "")
        blocks.append(
            f"[{doc_id}] title={title} path={metadata.get('path', '')}\n"
            f"{chunk.get('content', '')}"
        )
    return "\n\n---\n\n".join(blocks)


def _best_sentences(query: str, chunks: list[dict], max_sentences: int = 3) -> list[str]:
    query_terms = set(tokenize(query))
    if _asks_for_specific_identity(query) and not _has_specific_identity_evidence(chunks):
        return []

    scored = []
    for idx, chunk in enumerate(chunks, start=1):
        text = _clean_context_text(chunk.get("content", ""))
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in sentences:
            clean = sentence.strip()
            if len(clean) < 30:
                continue
            if _looks_like_metadata(clean):
                continue
            overlap = len(query_terms & set(tokenize(clean)))
            if overlap:
                scored.append((overlap, clean, citation_for(chunk, idx)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [f"{sentence} {citation}" for _, sentence, citation in scored[:max_sentences]]


def _clean_context_text(text: str) -> str:
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and ":" in stripped:
            continue
        cleaned_lines.append(stripped)
    cleaned = " ".join(cleaned_lines)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _looks_like_metadata(sentence: str) -> bool:
    lowered = sentence.lower()
    return (
        lowered.startswith("source:")
        or lowered.startswith("year:")
        or lowered.startswith("type:")
        or " - source:" in lowered
        or " - type:" in lowered
    )


def _asks_for_specific_identity(query: str) -> bool:
    normalized = strip_accents(query.lower())
    identity_terms = ["nghe si nao", "ai dung", "ai su dung", "ten nghe si", "danh tinh"]
    return any(term in normalized for term in identity_terms)


def _has_specific_identity_evidence(chunks: list[dict]) -> bool:
    text = strip_accents(" ".join(chunk.get("content", "") for chunk in chunks).lower())
    # The lab corpus only contains anonymized/sample identities such as
    # "nghe si A"; treat that as insufficient for a real identity question.
    if "nghe si a" in text:
        return False
    return bool(re.search(r"\bnghe si\s+[a-zA-ZÀ-ỹ]{2,}", text))


def generate_with_citation(query: str, context_chunks: list[dict] | None = None) -> str:
    # top_k=5 leaves room for instruction, question and output while still
    # giving enough evidence for citation in this lightweight lab setup.
    chunks = context_chunks if context_chunks is not None else retrieve(query, top_k=5)
    chunks = reorder_for_llm(chunks)
    clarification = _clarification_question(query)
    if clarification:
        return clarification
    if not chunks:
        return _format_abstention("Không tìm thấy context phù hợp trong corpus đã ingest.")

    openai_answer = _openai_generate(query, chunks)
    if openai_answer:
        return openai_answer

    answer_sentences = _best_sentences(query, chunks)
    if not answer_sentences:
        return _format_abstention("Context hiện có không đủ bằng chứng để trả lời chắc chắn.")

    return _format_grounded_answer(answer_sentences, chunks)


def _openai_generate(query: str, chunks: list[dict]) -> str | None:
    """Generate a grounded answer with OpenAI when OPENAI_API_KEY is configured."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None


def _clarification_question(query: str) -> str | None:
    normalized = strip_accents(query.strip().lower())
    vague_questions = {
        "ma tuy",
        "nghe si",
        "phap luat ma tuy",
        "cai nghien",
        "tin tuc ma tuy",
    }
    if normalized in vague_questions or len(tokenize(query)) <= 1:
        return (
            "Bạn muốn hỏi cụ thể về nội dung nào: quy định pháp luật, tin tức đã ingest, "
            "hay quy trình hỗ trợ/cai nghiện?"
        )
    return None


def _format_grounded_answer(answer_sentences: list[str], chunks: list[dict]) -> str:
    sources = []
    seen = set()
    for index, chunk in enumerate(chunks, start=1):
        citation = citation_for(chunk, index)
        if citation in seen:
            continue
        seen.add(citation)
        metadata = chunk.get("metadata", {})
        title = metadata.get("title") or metadata.get("source") or citation.strip("[]")
        sources.append(f"{citation} {title}")
        if len(sources) >= 3:
            break

    return (
        "Trả lời ngắn gọn: "
        + " ".join(answer_sentences)
        + "\n\nBằng chứng:\n- "
        + "\n- ".join(sources)
        + "\n\nGiới hạn: Câu trả lời chỉ dựa trên corpus đã ingest; nếu cần kết luận pháp lý chính thức, hãy kiểm tra văn bản gốc."
    )


def _format_abstention(reason: str) -> str:
    return (
        "Trả lời ngắn gọn: I cannot verify this information"
        f"\n\nBằng chứng: {reason}"
        "\n\nGiới hạn: Tôi không suy đoán ngoài context RAG hiện có. Bạn có thể hỏi cụ thể hơn hoặc bổ sung tài liệu nguồn."
    )

    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip()
    try:
        client = OpenAI(api_key=api_key)
        context = format_context(chunks)
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "<context>\n"
                        f"{context}\n"
                        "</context>\n\n"
                        "<question>\n"
                        f"{query}\n"
                        "</question>"
                    ),
                },
            ],
        )
        answer = response.choices[0].message.content or ""
        return answer.strip() or None
    except Exception:
        return None


def answer_question(query: str, top_k: int = 5) -> dict:
    sources = retrieve(query, top_k=top_k)
    return {
        "answer": generate_with_citation(query, sources),
        "sources": sources,
        "context": format_context(sources),
    }


def main(query: str = "Luật phòng chống ma túy quy định gì?") -> str:
    return generate_with_citation(query)


if __name__ == "__main__":
    print(main())
