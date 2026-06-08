# Day 08 Group Project - Drug Law RAG

## Product

The group artifact combines:

1. A Streamlit RAG chatbot with citations and source display.
2. A deterministic evaluation pipeline with a 15-question golden dataset.
3. A/B comparison between dense-only and hybrid retrieval.

## Architecture

```text
Documents
  -> Markdown standardization
  -> Chunking
  -> Local index / optional Weaviate

Question
  -> TF-IDF semantic retrieval
  -> BM25 lexical retrieval
  -> RRF fusion
  -> Jina reranking when available
  -> Context reordering and formatting
  -> OpenAI generation when quota is available
  -> Extractive grounded fallback
  -> Answer + citations + sources
```

## Answer Policy

The chatbot uses a grounded system prompt in `src/task10_generation.py`.

Answer rules:

- Answer only from retrieved context.
- Cite factual claims with `[doc_id]`.
- Do not guess missing facts or identify a person unless the context explicitly states it.
- If evidence is insufficient, answer `I cannot verify this information`.
- If the question is too broad or ambiguous, ask one short clarification question.

Default answer format:

```text
Trả lời ngắn gọn: ...

Bằng chứng:
- [doc_id] source title

Giới hạn: ...
```

## API and Fallback Matrix

| Component | Preferred | Fallback used by project |
|---|---|---|
| Embeddings | OpenAI embeddings | Jina embeddings, then local TF-IDF |
| Vector store | Weaviate | Local `data/index/chunks.json` |
| Reranking | Jina Reranker | Local query-term overlap |
| Generation | OpenAI chat model | Extractive grounded answer |
| PageIndex | PageIndex API | Local vectorless keyword fallback |
| Evaluation | RAGAS/DeepEval | Deterministic heuristic metrics |

The project must not claim that a fallback is the external service itself.

## Run

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run individual tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
```

Run the chatbot:

```powershell
.\.venv\Scripts\python.exe -m streamlit run group_project\app.py
```

Run A/B evaluation:

```powershell
.\.venv\Scripts\python.exe group_project\evaluation\eval_pipeline.py
```

Outputs:

- `group_project/evaluation/results.json`
- `group_project/evaluation/results.md`

## Evaluation

Golden dataset:

- 15 questions.
- Legal, news, mixed, exact-term and abstention categories.
- Includes expected answer and expected source IDs.

Metrics:

- Context Recall: proportion of expected sources retrieved.
- Context Precision: proportion of retrieved chunks from expected sources.
- Faithfulness: lexical support from retrieved context plus valid citation coverage.
- Answer Relevance: token-level F1 against the expected answer.

These are transparent heuristic metrics. They make local evaluation and CI
possible without an LLM judge. RAGAS/DeepEval should be added when judge quota
is available.

## A/B Configurations

```text
Config A: Dense-only TF-IDF
Config B: TF-IDF + BM25 + RRF
```

The expected benefit of Config B is better recall for exact identifiers,
document numbers, names and legal terms.

## Merge Guidance

No merge is needed when all work is developed on the current branch.

Merge is needed only when another member has code on a different branch:

```powershell
git fetch
git merge <member-branch>
```

Before merging:

1. Commit or stash the current work.
2. Confirm ownership of conflicting files.
3. Run individual tests and group evaluation after resolving conflicts.

## Work Allocation Template

| Member | Responsibility | Main files |
|---|---|---|
| Member 1 | Data and conversion | `task1`, `task2`, `task3` |
| Member 2 | Retrieval and reranking | `task4` to `task9` |
| Member 3 | Generation and chatbot | `task10`, `group_project/app.py` |
| Member 4 | Evaluation and reporting | `group_project/evaluation/` |
