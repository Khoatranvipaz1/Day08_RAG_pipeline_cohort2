"""Lightweight A/B evaluation for the Day 08 RAG group project.

The evaluator is deterministic and does not require an LLM judge. It reports
the four metrics requested by the lab and clearly labels them as heuristics.
DeepEval/RAGAS can replace these functions later without changing the dataset.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag_utils import citation_for, tokenize
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import reciprocal_rank_fusion

DATASET_PATH = Path(__file__).with_name("golden_dataset.json")
RESULTS_JSON_PATH = Path(__file__).with_name("results.json")
RESULTS_MD_PATH = Path(__file__).with_name("results.md")
ABSTAIN_TEXT = "I cannot verify this information"


def load_golden_dataset(path: Path = DATASET_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def retrieve_dense(query: str, top_k: int = 5) -> list[dict]:
    return semantic_search(query, top_k=top_k)


def retrieve_hybrid(query: str, top_k: int = 5) -> list[dict]:
    dense = semantic_search(query, top_k=max(10, top_k * 2))
    sparse = lexical_search(query, top_k=max(10, top_k * 2))
    return reciprocal_rank_fusion([dense, sparse], top_k=top_k)


def source_ids(contexts: list[dict]) -> set[str]:
    return {
        str(item.get("metadata", {}).get("source", "")).strip()
        for item in contexts
        if item.get("metadata", {}).get("source")
    }


def grounded_answer(question: str, contexts: list[dict], should_abstain: bool) -> str:
    if should_abstain or not contexts:
        return ABSTAIN_TEXT

    query_terms = set(tokenize(question))
    candidates = []
    for index, chunk in enumerate(contexts, start=1):
        for sentence in re.split(r"(?<=[.!?])\s+", chunk.get("content", "")):
            sentence = sentence.strip()
            if len(sentence) < 25:
                continue
            overlap = len(query_terms & set(tokenize(sentence)))
            if overlap:
                candidates.append((overlap, sentence, citation_for(chunk, index)))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return ABSTAIN_TEXT
    return " ".join(f"{sentence} {citation}" for _, sentence, citation in candidates[:3])


def context_recall(expected_sources: list[str], contexts: list[dict], should_abstain: bool) -> float:
    if should_abstain:
        return 1.0
    expected = set(expected_sources)
    if not expected:
        return 1.0
    return len(expected & source_ids(contexts)) / len(expected)


def context_precision(expected_sources: list[str], contexts: list[dict], should_abstain: bool) -> float:
    if should_abstain:
        return 1.0
    if not contexts:
        return 0.0
    expected = set(expected_sources)
    relevant = sum(
        1 for item in contexts
        if item.get("metadata", {}).get("source") in expected
    )
    return relevant / len(contexts)


def answer_relevance(expected_answer: str, actual_answer: str, should_abstain: bool) -> float:
    if should_abstain:
        return 1.0 if ABSTAIN_TEXT.lower() in actual_answer.lower() else 0.0
    expected_terms = set(tokenize(expected_answer))
    actual_terms = set(tokenize(actual_answer))
    if not expected_terms or not actual_terms:
        return 0.0
    precision = len(expected_terms & actual_terms) / len(actual_terms)
    recall = len(expected_terms & actual_terms) / len(expected_terms)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def faithfulness(actual_answer: str, contexts: list[dict], should_abstain: bool) -> float:
    if should_abstain:
        return 1.0 if ABSTAIN_TEXT.lower() in actual_answer.lower() else 0.0
    if not contexts or actual_answer == ABSTAIN_TEXT:
        return 0.0
    context_terms = set(tokenize(" ".join(item.get("content", "") for item in contexts)))
    answer_terms = set(tokenize(re.sub(r"\[[^\]]+\]", "", actual_answer)))
    lexical_support = len(answer_terms & context_terms) / max(len(answer_terms), 1)
    valid_citations = sum(
        1 for source in source_ids(contexts)
        if f"[{source}]" in actual_answer
    )
    citation_score = min(1.0, valid_citations / max(1, len(contexts)))
    return 0.8 * lexical_support + 0.2 * citation_score


def evaluate_config(name: str, retriever, dataset: list[dict], top_k: int = 5) -> dict:
    rows = []
    for item in dataset:
        contexts = retriever(item["question"], top_k=top_k)
        answer = grounded_answer(item["question"], contexts, item.get("should_abstain", False))
        metrics = {
            "context_recall": context_recall(
                item.get("expected_context", []), contexts, item.get("should_abstain", False)
            ),
            "context_precision": context_precision(
                item.get("expected_context", []), contexts, item.get("should_abstain", False)
            ),
            "faithfulness": faithfulness(answer, contexts, item.get("should_abstain", False)),
            "answer_relevance": answer_relevance(
                item["expected_answer"], answer, item.get("should_abstain", False)
            ),
        }
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "category": item["category"],
                "answer": answer,
                "retrieved_sources": sorted(source_ids(contexts)),
                "metrics": metrics,
            }
        )

    summary = {
        metric: mean(row["metrics"][metric] for row in rows)
        for metric in ["context_recall", "context_precision", "faithfulness", "answer_relevance"]
    }
    return {"name": name, "summary": summary, "rows": rows}


def diagnosis(row: dict) -> str:
    metrics = row["metrics"]
    recall = metrics["context_recall"]
    faith = metrics["faithfulness"]
    relevance = metrics["answer_relevance"]
    if recall < 0.5:
        return "Retrieval/Indexing"
    if faith < 0.7:
        return "Generation/Grounding"
    if relevance < 0.35:
        return "Augmentation/Answer relevance"
    if metrics["context_precision"] < 0.25:
        return "Retrieval noise/Reranking"
    return "OK"


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# RAG Evaluation Results",
        "",
        "## Setup",
        "",
        "- Golden dataset: 15 questions",
        "- Config A: Dense-only TF-IDF retrieval",
        "- Config B: Hybrid TF-IDF + BM25 + RRF",
        "- Metrics: heuristic Context Recall, Context Precision, Faithfulness, Answer Relevance",
        "- Note: heuristic evaluation is deterministic; RAGAS/DeepEval can replace it when an LLM judge quota is available.",
        "",
        "## Overall Scores",
        "",
        "| Config | Context Recall | Context Precision | Faithfulness | Answer Relevance |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        s = result["summary"]
        lines.append(
            f"| {result['name']} | {s['context_recall']:.3f} | "
            f"{s['context_precision']:.3f} | {s['faithfulness']:.3f} | "
            f"{s['answer_relevance']:.3f} |"
        )

    lines.extend(["", "## Worst Performers", ""])
    for result in results:
        worst = sorted(
            result["rows"],
            key=lambda row: mean(row["metrics"].values()),
        )[:5]
        lines.extend(
            [
                f"### {result['name']}",
                "",
                "| ID | Recall | Precision | Faithfulness | Relevance | Diagnosis |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in worst:
            m = row["metrics"]
            lines.append(
                f"| {row['id']} | {m['context_recall']:.2f} | "
                f"{m['context_precision']:.2f} | {m['faithfulness']:.2f} | "
                f"{m['answer_relevance']:.2f} | {diagnosis(row)} |"
            )
        lines.append("")

    dense, hybrid = results
    delta = {
        key: hybrid["summary"][key] - dense["summary"][key]
        for key in dense["summary"]
    }
    lines.extend(
        [
            "## A/B Analysis",
            "",
            f"- Context Recall delta (Hybrid - Dense): `{delta['context_recall']:+.3f}`",
            f"- Context Precision delta: `{delta['context_precision']:+.3f}`",
            f"- Faithfulness delta: `{delta['faithfulness']:+.3f}`",
            f"- Answer Relevance delta: `{delta['answer_relevance']:+.3f}`",
            "- Hybrid is expected to help exact identifiers such as `105/2021/NĐ-CP` because BM25 complements semantic retrieval.",
            "- Low Context Recall indicates retrieval/indexing work; low Faithfulness indicates generation/grounding work.",
            "",
            "## Next Actions",
            "",
            "1. Replace sample documents with authoritative legal PDFs and real news articles.",
            "2. Run RAGAS or DeepEval when an LLM-judge quota is available.",
            "3. Tune `top_k`, RRF constant and Jina reranking using the worst-performing questions.",
            "4. Keep abstention tests in CI to prevent unsupported legal claims and PII leakage.",
            "",
        ]
    )
    return "\n".join(lines)


def run_evaluation(top_k: int = 3) -> list[dict]:
    dataset = load_golden_dataset()
    results = [
        evaluate_config("Dense-only", retrieve_dense, dataset, top_k=top_k),
        evaluate_config("Hybrid BM25 + TF-IDF + RRF", retrieve_hybrid, dataset, top_k=top_k),
    ]
    RESULTS_JSON_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    RESULTS_MD_PATH.write_text(render_markdown(results), encoding="utf-8")
    return results


if __name__ == "__main__":
    evaluated = run_evaluation()
    for config in evaluated:
        print(config["name"], config["summary"])
