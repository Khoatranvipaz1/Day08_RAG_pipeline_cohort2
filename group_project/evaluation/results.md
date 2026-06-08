# RAG Evaluation Results

## Setup

- Golden dataset: 15 questions
- Config A: Dense-only TF-IDF retrieval
- Config B: Hybrid TF-IDF + BM25 + RRF
- Metrics: heuristic Context Recall, Context Precision, Faithfulness, Answer Relevance
- Note: heuristic evaluation is deterministic; RAGAS/DeepEval can replace it when an LLM judge quota is available.

## Overall Scores

| Config | Context Recall | Context Precision | Faithfulness | Answer Relevance |
|---|---:|---:|---:|---:|
| Dense-only | 0.867 | 0.400 | 0.991 | 0.174 |
| Hybrid BM25 + TF-IDF + RRF | 0.867 | 0.400 | 0.991 | 0.174 |

## Worst Performers

### Dense-only

| ID | Recall | Precision | Faithfulness | Relevance | Diagnosis |
|---|---:|---:|---:|---:|---|
| news_02 | 0.00 | 0.00 | 1.00 | 0.02 | Retrieval/Indexing |
| legal_03 | 0.00 | 0.00 | 1.00 | 0.05 | Retrieval/Indexing |
| news_01 | 1.00 | 0.33 | 0.93 | 0.02 | Augmentation/Answer relevance |
| news_03 | 1.00 | 0.33 | 0.93 | 0.05 | Augmentation/Answer relevance |
| news_04 | 1.00 | 0.33 | 1.00 | 0.00 | Augmentation/Answer relevance |

### Hybrid BM25 + TF-IDF + RRF

| ID | Recall | Precision | Faithfulness | Relevance | Diagnosis |
|---|---:|---:|---:|---:|---|
| news_02 | 0.00 | 0.00 | 1.00 | 0.02 | Retrieval/Indexing |
| legal_03 | 0.00 | 0.00 | 1.00 | 0.05 | Retrieval/Indexing |
| news_01 | 1.00 | 0.33 | 0.93 | 0.02 | Augmentation/Answer relevance |
| news_03 | 1.00 | 0.33 | 0.93 | 0.05 | Augmentation/Answer relevance |
| news_04 | 1.00 | 0.33 | 1.00 | 0.00 | Augmentation/Answer relevance |

## A/B Analysis

- Context Recall delta (Hybrid - Dense): `+0.000`
- Context Precision delta: `+0.000`
- Faithfulness delta: `+0.000`
- Answer Relevance delta: `+0.000`
- Hybrid is expected to help exact identifiers such as `105/2021/NĐ-CP` because BM25 complements semantic retrieval.
- Low Context Recall indicates retrieval/indexing work; low Faithfulness indicates generation/grounding work.

## Next Actions

1. Replace sample documents with authoritative legal PDFs and real news articles.
2. Run RAGAS or DeepEval when an LLM-judge quota is available.
3. Tune `top_k`, RRF constant and Jina reranking using the worst-performing questions.
4. Keep abstention tests in CI to prevent unsupported legal claims and PII leakage.
