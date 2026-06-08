"""Task 10: grounded generation with citation."""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from .rag_utils import citation_for, tokenize
from .task9_retrieval_pipeline import retrieve

SYSTEM_PROMPT = """Answer only from the provided context.
Cite sources using [doc_id] when possible.
If the context is insufficient, say 'I cannot verify this information'."""


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
    scored = []
    for idx, chunk in enumerate(chunks, start=1):
        sentences = re.split(r"(?<=[.!?])\s+", chunk.get("content", ""))
        for sentence in sentences:
            clean = sentence.strip()
            if len(clean) < 30:
                continue
            overlap = len(query_terms & set(tokenize(clean)))
            if overlap:
                scored.append((overlap, clean, citation_for(chunk, idx)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [f"{sentence} {citation}" for _, sentence, citation in scored[:max_sentences]]


def generate_with_citation(query: str, context_chunks: list[dict] | None = None) -> str:
    # top_k=5 leaves room for instruction, question and output while still
    # giving enough evidence for citation in this lightweight lab setup.
    chunks = context_chunks if context_chunks is not None else retrieve(query, top_k=5)
    chunks = reorder_for_llm(chunks)
    if not chunks:
        return "I cannot verify this information"

    openai_answer = _openai_generate(query, chunks)
    if openai_answer:
        return openai_answer

    answer_sentences = _best_sentences(query, chunks)
    if not answer_sentences:
        return "I cannot verify this information"

    return " ".join(answer_sentences)


def _openai_generate(query: str, chunks: list[dict]) -> str | None:
    """Generate a grounded answer with OpenAI when OPENAI_API_KEY is configured."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

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
