# Day 08 RAG Pipeline - Checklist Không Bỏ Sót

File này gom lại toàn bộ việc cần làm cho lab Day 08: công cụ cần dùng, pipeline R-A-G, từng task cá nhân, bài nhóm, cách đo RAG, cách tính eval, A/B test, CI/CD và checklist rà soát cuối.

---

## 0. Mục Tiêu Cần Nhớ

Day 08 nâng cấp từ Retrieval Pipeline thành RAG pipeline hoàn chỉnh:

```text
R = Retrieval/Search + Rerank
A = Augmentation/đóng gói context
G = Generation/sinh câu trả lời grounded có citation
```

Pipeline tổng quan:

```text
Offline:
Documents -> Ingestion -> Vector Store

Runtime:
Question -> Retrieval -> Augmentation -> Generation -> Answer
```

Mục tiêu cuối:

- Thu thập dữ liệu pháp luật và báo chí.
- Convert dữ liệu sang Markdown.
- Chunking + indexing vào vector store.
- Có dense search, lexical/BM25 search, hybrid search.
- Có rerank/fusion.
- Có fallback PageIndex/vectorless.
- Sinh câu trả lời grounded, có citation.
- Có evaluation bằng RAGAS/DeepEval/TruLens.
- Có báo cáo A/B và chẩn đoán lỗi theo RAG Triad.

---

## 1. Công Cụ Cần Dùng

### Bản thay thế đã dùng trong repo này

Do môi trường hiện tại dùng Python 3.14 trên Windows, một số package full trong README dễ lỗi khi cài, đặc biệt `crawl4ai` kéo theo `lxml` và có thể phải build native. Vì vậy `requirements.txt` đã được đổi sang bản nhẹ hơn để làm lab chạy được trước.

| Theo README/full stack | Đã thay bằng trong `requirements.txt` | Dùng cho task | Ghi chú |
|---|---|---|---|
| `crawl4ai` | `requests` + `beautifulsoup4` + `html5lib` | Task 2 | Crawl đơn giản, parse HTML và lưu metadata. Nếu cần crawler mạnh thì cài `crawl4ai` riêng sau. |
| `sentence-transformers` | `openai` embeddings hoặc `scikit-learn` TF-IDF/local vectors | Task 4, 5 | Tránh kéo `torch`. Nếu cần local embedding thật thì cài `sentence-transformers` riêng. |
| `weaviate-client` | `scikit-learn` + local in-memory/file index | Task 4, 5, 9 | Đủ demo retrieval nhẹ. Nếu cần vector DB/hybrid built-in thì cài Weaviate sau. |
| `pageindex` | Mock/vectorless fallback tự viết | Task 8, 9 | Function vẫn giữ đúng signature, nhưng fallback có thể đọc local corpus/search heuristic. |
| `deepeval` | Heuristic eval + `pandas` report | Group evaluation | Có thể chấm context recall/citation/keyword/abstention trước. Nếu có API key và môi trường ổn thì cài `deepeval` hoặc `ragas`. |
| `gradio`/`chainlit` | `streamlit` | Group chatbot | Chỉ giữ một UI để giảm dependency. |
| `chromadb`/`faiss-cpu` | `scikit-learn` | Optional vector store | Có thể thêm sau nếu cần persistent vector store. |

Quy tắc khi báo cáo/demo:

- Nói rõ đây là **lightweight implementation** để chạy ổn trên Windows/Python 3.14.
- API/function vẫn nên giữ đúng signature trong README để test dễ kiểm tra.
- Các phần bị thay thế nên được ghi là fallback/mock/local implementation, không ghi nhầm là đã dùng Weaviate/PageIndex/DeepEval thật.

### Thu thập và chuẩn hóa dữ liệu

| Việc | Công cụ gợi ý | Ghi chú |
|---|---|---|
| Tải văn bản pháp luật | Trình duyệt, nguồn pháp luật chính thống | Lưu file gốc PDF/DOCX |
| Crawl bài báo | Crawl4AI | Lưu HTML/JSON/Markdown + metadata |
| Convert PDF/DOCX/HTML sang Markdown | MarkItDown | Output vào `data/standardized/` |

### Chunking, embedding, vector store

| Việc | Công cụ gợi ý | Ghi chú |
|---|---|---|
| Chunking | `langchain-text-splitters` | `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter` |
| Embedding nhẹ | `sentence-transformers/all-MiniLM-L6-v2` | Nhanh, nhẹ |
| Embedding tiếng Việt/multilingual | `BAAI/bge-m3` | Phù hợp corpus tiếng Việt |
| Embedding API | OpenAI `text-embedding-3-small` | Cần API key |
| Vector store | Weaviate, ChromaDB, FAISS | Weaviate mạnh về hybrid search |

### Retrieval, rerank, fallback

| Việc | Công cụ gợi ý | Ghi chú |
|---|---|---|
| Dense search | Vector store similarity search | Semantic retrieval |
| Lexical search | `rank-bm25` | Bắt keyword, tên riêng, mã văn bản |
| Fusion | RRF hoặc alpha weighting | Gộp dense + sparse |
| Rerank | Jina Reranker, Cohere Rerank, Qwen Reranker, MMR | Re-score candidates |
| Vectorless fallback | PageIndex | Dùng khi hybrid không đủ tốt |

### Generation và evaluation

| Việc | Công cụ gợi ý | Ghi chú |
|---|---|---|
| LLM generation | OpenAI, Gemini, local LLM | Trả lời có citation |
| RAG evaluation | RAGAS, DeepEval, TruLens | Faithfulness, relevance, recall, precision |
| Chat UI | Streamlit, Gradio, Chainlit | Bài nhóm chatbot |
| CI/CD eval | GitHub Actions | Block deploy nếu metric thấp |

---

## 2. Cấu Trúc Thư Mục Cần Có

```text
data/
  landing/
    legal/
    news/
  standardized/
    legal/
    news/

src/
  task1_collect_legal_docs.py
  task2_crawl_news.py
  task3_convert_markdown.py
  task4_chunking_indexing.py
  task5_semantic_search.py
  task6_lexical_search.py
  task7_reranking.py
  task8_pageindex_vectorless.py
  task9_retrieval_pipeline.py
  task10_generation.py

group_project/
  README.md
  evaluation/
    golden_dataset.json
    eval_pipeline.py
    results.md
```

---

## 3. Quy Trình Cá Nhân Task 1-10

### Task 1 - Thu thập văn bản pháp luật

Mục tiêu:

- Tải tối thiểu 3 văn bản pháp luật về ma túy/chất cấm.
- Lưu file gốc vào `data/landing/legal/`.

Gợi ý tài liệu:

- Luật Phòng, chống ma túy 2021.
- Nghị định 105/2021/NĐ-CP.
- Bộ luật Hình sự 2015 sửa đổi 2017, phần tội phạm về ma túy.

Checklist:

- [ ] Có ít nhất 3 file PDF/DOCX.
- [ ] Tên file rõ ràng, ví dụ `luat-phong-chong-ma-tuy-2021.pdf`.
- [ ] File nằm đúng `data/landing/legal/`.

### Task 2 - Crawl bài báo

Mục tiêu:

- Crawl tối thiểu 5 bài báo về nghệ sĩ Việt Nam liên quan tới ma túy.
- Lưu vào `data/landing/news/`.

Metadata nên có:

- URL gốc.
- Ngày crawl.
- Tiêu đề.
- Nội dung.
- Nguồn báo.

Checklist:

- [ ] Có ít nhất 5 file tin tức.
- [ ] Mỗi file có metadata.
- [ ] Lưu đúng `data/landing/news/`.
- [ ] Nội dung đủ dài để dùng làm evidence.

### Task 3 - Convert sang Markdown

Mục tiêu:

- Convert toàn bộ file trong `data/landing/` sang Markdown.
- Output vào `data/standardized/`.
- Giữ cấu trúc `legal/`, `news/`.

Tool:

```bash
pip install markitdown
```

Checklist:

- [ ] Legal docs có bản `.md`.
- [ ] News docs có bản `.md`.
- [ ] Không mất metadata quan trọng.
- [ ] Markdown đọc được, không toàn ký tự lỗi.

### Task 4 - Chunking và Indexing

Mục tiêu:

- Đọc Markdown.
- Chunk văn bản.
- Tạo embedding.
- Index vào vector store.

Quyết định cần ghi rõ trong code:

- Dùng splitter nào.
- `chunk_size` bao nhiêu.
- `chunk_overlap` bao nhiêu.
- Embedding model nào.
- Dimension bao nhiêu.
- Vì sao chọn.

Gợi ý:

```text
Chunking: RecursiveCharacterTextSplitter
chunk_size: 800-1200 tokens/chars tùy implementation
overlap: 100-200
Embedding: BAAI/bge-m3 nếu ưu tiên tiếng Việt
Vector store: ChromaDB/FAISS nếu cần đơn giản, Weaviate nếu muốn hybrid built-in
```

Checklist:

- [ ] Toàn bộ Markdown được load.
- [ ] Chunk có `content`.
- [ ] Chunk có metadata: source, path, doc_type, title, date nếu có.
- [ ] Vector store có dữ liệu.
- [ ] Có thể reload index sau khi chạy lại.

### Task 5 - Semantic Search

Mục tiêu:

```python
def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    ...
```

Output format:

```python
{
    "content": "...",
    "score": 0.85,
    "metadata": {...}
}
```

Checklist:

- [ ] Nhận query string.
- [ ] Trả về list dict.
- [ ] Có `content`, `score`, `metadata`.
- [ ] Sorted theo score giảm dần.
- [ ] Hoạt động với embedding model đã index.

### Task 6 - Lexical Search/BM25

Mục tiêu:

```python
def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    ...
```

Tool:

```bash
pip install rank-bm25
```

Vì sao cần BM25:

- Dense search mạnh về ngữ nghĩa.
- BM25 mạnh về exact keyword, mã lỗi, tên riêng, số ticket, điều luật.
- Hybrid thường tăng Context Recall rất mạnh.

Checklist:

- [ ] Corpus được tokenized.
- [ ] Query được tokenized cùng cách với corpus.
- [ ] Trả đúng format `content`, `score`, `metadata`.
- [ ] Sorted theo score giảm dần.

### Task 7 - Fusion và Reranking

Mục tiêu:

```python
def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    ...
```

Phương án dễ làm:

- Dùng RRF để gộp dense + BM25.
- Sau đó rerank top candidates nếu có model/API.

RRF:

```text
RRF(d) = sum over rankers: 1 / (k + rank_r(d))
```

Gợi ý `k = 60`.

Checklist:

- [ ] Merge được kết quả semantic và lexical.
- [ ] Deduplicate theo content/source/chunk_id.
- [ ] Có score hợp nhất.
- [ ] Rerank/re-sort candidates.
- [ ] Trả top_k.

### Task 8 - PageIndex Vectorless Fallback

Mục tiêu:

```python
def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    ...
```

Vai trò:

- Fallback khi hybrid search không có kết quả đủ tốt.
- Dùng cho trường hợp vector index lỗi, thiếu, hoặc score thấp.

Checklist:

- [ ] Có function đúng signature.
- [ ] Trả đúng format.
- [ ] Có xử lý khi thiếu API key/config.
- [ ] Không làm vỡ pipeline nếu PageIndex chưa sẵn sàng.

### Task 9 - Retrieval Pipeline Hoàn Chỉnh

Mục tiêu:

```python
def retrieve(query: str, top_k: int = 5, score_threshold: float = 0.3) -> list[dict]:
    ...
```

Luồng:

```text
Query
  -> semantic_search
  -> lexical_search
  -> merge/fusion bằng RRF hoặc alpha weighting
  -> rerank
  -> nếu top score < threshold thì fallback PageIndex
  -> return top_k
```

Checklist:

- [ ] Chạy semantic search.
- [ ] Chạy lexical search.
- [ ] Gộp kết quả.
- [ ] Rerank.
- [ ] Có fallback theo threshold.
- [ ] Return format thống nhất.
- [ ] Có metadata đủ để citation.

### Task 10 - Generation Có Citation

Mục tiêu:

```python
def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    ...

def generate_with_citation(query: str, context_chunks: list[dict]) -> str:
    ...
```

Yêu cầu chính:

- Reorder context để giảm "lost in the middle".
- Format context rõ ràng.
- Tách system, context/evidence, question.
- Chỉ trả lời dựa trên context.
- Có citation.
- Nếu thiếu evidence thì trả: `I cannot verify this information`.

Prompt pattern gợi ý:

```text
<system>
You are a grounded RAG assistant.
Answer only from the provided context.
Cite sources using [doc_id] when possible.
If evidence is insufficient, say "I cannot verify this information".
</system>

<context>
[doc_1] source=..., title=..., date=...
...
</context>

<question>
...
</question>
```

Checklist:

- [ ] Context có source/citation id.
- [ ] Không nhét context thô lộn xộn.
- [ ] Context không vượt token budget.
- [ ] Có abstention khi thiếu evidence.
- [ ] Output có citation.
- [ ] Có comment giải thích `top_k`, `top_p` nếu dùng.

---

## 4. R-A-G Pipeline Đúng Theo Slide

### R - Retrieval

Kỹ thuật chính:

- Dense Search/Semantic Search.
- Sparse Search/BM25/TF-IDF.
- Hybrid Search.
- RRF.
- Reranking.
- MMR.
- Pre-filtering bằng metadata.
- Query routing.
- Multi-query.
- Parent-child retrieval.
- Query expansion.
- Query decomposition.
- Step-back prompting.
- HyDE.

Khi nào dùng:

- Corpus có tên riêng, số hiệu, điều luật, mã lỗi -> thêm BM25.
- Query ngắn/mơ hồ -> multi-query hoặc HyDE.
- Query nhiều ý/multi-hop -> decomposition.
- Index quá lớn/nhiễu -> pre-filter metadata.

### A - Augmentation

Kỹ thuật chính:

- Context injection.
- Document reordering.
- Instruction tuning.
- Metadata integration.
- Citation formatting.
- Context compression.
- Token budget management.
- Conflict resolution.
- Grounding constraints.
- Context deduplication.

Checklist augmentation:

- [ ] Context có source.
- [ ] Context có metadata cần thiết.
- [ ] Context được reorder.
- [ ] Context không quá dài.
- [ ] Context không trùng lặp quá nhiều.
- [ ] Có rule xử lý conflict.
- [ ] Prompt tách rõ system/context/question.

### G - Generation

Kỹ thuật chính:

- Grounded generation.
- Self-correction/self-check.
- Output formatting.
- Citation generation.
- Abstention.
- Chain-of-thought nội bộ nếu cần, nhưng không bắt buộc hiển thị.
- LLM selection/routing.
- Safety & PII filtering.
- Streaming generation.

Checklist generation:

- [ ] Chỉ trả lời theo context.
- [ ] Có citation cho claim quan trọng.
- [ ] Không bịa nếu thiếu evidence.
- [ ] Có format Markdown/Table/JSON nếu bài yêu cầu.
- [ ] Kiểm tra dữ liệu nhạy cảm/PII/access level.

---

## 5. Pipeline Đo Lường RAG

Không chấm RAG bằng một con số duy nhất. Phải tách lỗi theo:

- Retriever.
- Augmentation.
- Generator.
- Indexing/data.

### RAG Triad

| Metric | Đo cái gì | Lỗi thường chỉ ra |
|---|---|---|
| Context Recall | Retriever có tìm đủ evidence không? | Retrieval/indexing |
| Faithfulness | Câu trả lời có bám context không? | Generation/prompt |
| Answer Relevance | Câu trả lời có đúng ý câu hỏi không? | Query understanding/augmentation/generation |
| Context Precision | Context lấy về có hữu ích không? | Retrieval noise/rerank |

### Chẩn đoán bằng bảng điểm

| Context Recall | Faithfulness | Answer Relevance | Chẩn đoán |
|---|---|---|---|
| Cao | Cao | Cao | Hệ thống hoạt động tốt |
| Thấp | Cao | Thấp | Sửa Retrieval/search |
| Cao | Thấp | Cao | Sửa Generation/prompt/grounding |
| Thấp | Thấp | Thấp | Sửa Indexing/data/chunking |
| Cao | Cao | Thấp | Sửa Augmentation/context packaging |

Ghi nhớ:

- Recall cao + Faithfulness thấp -> tìm đúng tài liệu nhưng model nói sai.
- Recall thấp + Faithfulness cao -> model ngoan nhưng retriever tìm thiếu.
- Tất cả thấp -> kiểm tra data, chunking, indexing.
- Recall cao + Faithfulness cao + Relevance thấp -> context đúng nhưng đóng gói/prompt sai.

---

## 6. Tính Eval Như Nào, Dùng Gì Tính

### Golden dataset

Tạo file:

```text
group_project/evaluation/golden_dataset.json
```

Tối thiểu 15 item. Mỗi item nên có:

```json
{
  "id": "q001",
  "question": "Câu hỏi cần kiểm tra",
  "expected_answer": "Ý chính câu trả lời đúng",
  "expected_context": [
    "Tên tài liệu hoặc chunk/source cần retrieve được"
  ],
  "category": "legal/news/mixed",
  "difficulty": "easy/medium/hard"
}
```

Nên chia bộ câu hỏi:

- 5 câu legal lookup trực tiếp.
- 5 câu news/factual lookup.
- 3 câu multi-hop/tổng hợp.
- 2 câu không đủ evidence để test abstention.

### Option A - RAGAS

Dùng khi muốn framework chuẩn cho RAG:

Metrics:

- `faithfulness`.
- `answer_relevancy`.
- `context_recall`.
- `context_precision`.

Data format:

```python
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}
```

Cách tính:

1. Với mỗi item trong golden dataset, gọi RAG pipeline.
2. Lưu câu trả lời vào `answer`.
3. Lưu retrieved chunks vào `contexts`.
4. Lưu expected answer vào `ground_truth`.
5. Gọi RAGAS evaluate.
6. Xuất bảng điểm trung bình và điểm từng câu.

### Option B - DeepEval

Dùng khi muốn dễ tích hợp pytest/CI:

Metrics:

- `FaithfulnessMetric`.
- `AnswerRelevancyMetric`.
- `ContextualRecallMetric`.
- `ContextualPrecisionMetric`.

Cách tính:

1. Tạo `LLMTestCase`.
2. `input` = question.
3. `actual_output` = answer từ RAG.
4. `expected_output` = expected answer.
5. `retrieval_context` = list content chunks.
6. Chạy `evaluate(test_cases, metrics)`.

### Option C - TruLens

Dùng khi muốn dashboard:

Feedback:

- Groundedness/Faithfulness.
- Input-output relevance.
- Context relevance.

Cách tính:

1. Wrap RAG app bằng `TruCustomApp`.
2. Instrument retrieval/generation.
3. Chạy golden dataset.
4. Xem dashboard.

### Nếu không có API key/eval framework

Vẫn nên có evaluation script đơn giản:

- Context Recall heuristic:
  - Kiểm tra retrieved source có nằm trong `expected_context` không.
- Citation coverage:
  - Kiểm tra answer có citation dạng `[...]` không.
- Abstention accuracy:
  - Câu không đủ evidence phải trả `I cannot verify this information`.
- Keyword relevance:
  - So khớp keyword chính giữa expected answer và actual answer.

Ghi rõ trong báo cáo đây là heuristic eval, không thay thế hoàn toàn RAGAS/DeepEval.

---

## 7. A/B Test Bắt Buộc Cho Nhóm

Cần so sánh ít nhất 2 config.

### Gợi ý A/B dễ làm nhất

```text
Config A: Dense-only
Config B: Hybrid = Dense + BM25 + RRF
```

Kỳ vọng:

- Dense-only có thể bỏ sót mã văn bản, tên riêng, số hiệu.
- Hybrid tăng Context Recall rõ nhất.

### Gợi ý A/B khác

```text
Config A: Hybrid without rerank
Config B: Hybrid with rerank
```

Hoặc:

```text
Config A: No context reordering
Config B: Reorder context to reduce lost-in-the-middle
```

### Bảng báo cáo nên có

| Config | Context Recall | Context Precision | Faithfulness | Answer Relevance | Nhận xét |
|---|---:|---:|---:|---:|---|
| Dense-only | | | | | |
| Hybrid + RRF | | | | | |

Phân tích bắt buộc:

- Config nào tốt hơn?
- Metric nào tăng/giảm?
- Query nào fail nặng nhất?
- Lỗi thuộc Retrieval, Generation, Indexing hay Augmentation?
- Cần sửa gì tiếp theo?

---

## 8. File `results.md` Nên Viết Gì

File:

```text
group_project/evaluation/results.md
```

Cấu trúc đề xuất:

```markdown
# RAG Evaluation Results

## Setup
- Dataset:
- Number of questions:
- Config A:
- Config B:
- Eval framework:
- Metrics:

## Overall Scores
| Config | Context Recall | Context Precision | Faithfulness | Answer Relevance |
|---|---:|---:|---:|---:|

## Worst Performers
| Question ID | Symptom | Metric failed | Diagnosis | Fix |
|---|---|---|---|---|

## RAG Triad Diagnosis
- Retrieval errors:
- Generation errors:
- Augmentation errors:
- Indexing/data errors:

## Conclusion
- Best config:
- Main improvement:
- Next actions:
```

---

## 9. CI/CD Cho RAG Evaluation

Tư duy:

```text
Code RAG != Code Web
Deploy RAG phải test hành vi AI
```

Luồng:

```text
Git Push -> RAGAS/DeepEval -> Pass thì Deploy -> Fail thì Block
```

Threshold gợi ý:

```text
faithfulness >= 0.80
answer_relevance >= 0.75
context_recall >= 0.75
context_precision >= 0.70
```

Checklist:

- [ ] Có script chạy eval tự động.
- [ ] Có threshold pass/fail.
- [ ] Nếu fail thì không deploy.
- [ ] Có log query nào fail.
- [ ] Có báo cáo worst performers.

---

## 10. Bài Nhóm

Sau bài cá nhân, nhóm cần có ít nhất:

### Option 1 - RAG Chatbot

Yêu cầu:

- UI bằng Streamlit/Gradio/Chainlit.
- Trả lời có citation.
- Hiển thị source documents.
- Có follow-up/conversation memory nếu làm bonus.

Luồng:

```text
UI -> retrieve() -> generate_with_citation() -> display answer + sources
```

Checklist:

- [ ] Gõ câu hỏi được.
- [ ] Có câu trả lời.
- [ ] Có citation.
- [ ] Có danh sách source.
- [ ] Source có score/metadata.
- [ ] Có xử lý khi không đủ evidence.

### Option 2 - RAG Evaluation Pipeline

Deliverables:

- [ ] `group_project/evaluation/golden_dataset.json` có 15+ Q&A.
- [ ] `group_project/evaluation/eval_pipeline.py` chạy được.
- [ ] `group_project/evaluation/results.md` có bảng điểm + phân tích.
- [ ] Có A/B ít nhất 2 config.
- [ ] Có 4 metrics: faithfulness, answer relevance, context recall, context precision.

---

## 11. Agentic RAG, MCP, A2A - Phần Chuẩn Bị Tiếp Theo

Khi single-agent RAG quá tải, tách thành nhiều worker.

Use case gợi ý:

```text
Drug Law RAG Assistant with Evaluation
```

Tách worker:

| Worker | Nhiệm vụ |
|---|---|
| Retriever Worker | Dense + BM25 + RRF/rerank |
| Generator Worker | Đóng gói context, sinh answer có citation |
| Evaluator Worker | Chấm RAGAS/DeepEval, chẩn đoán lỗi |
| Tool/Fallback Worker | PageIndex, database/API, web/API ngoài nếu có |

Artifact Day 08 có thể thành worker:

- `task5`, `task6`, `task7`, `task9` -> Retriever Worker.
- `task10` -> Generator Worker.
- `eval_pipeline.py` -> Evaluator Worker.
- `task8` -> Fallback Tool Worker.

Ý chính:

- MCP: chuẩn để agent gọi tool/data source.
- A2A: chuẩn để agent giao tiếp với agent khác.
- Multi-agent nên chia theo tool, domain hoặc bước xử lý.

---

## 12. Checklist Rà Soát 4 Vòng

### Vòng 1 - Rà README/task cá nhân

- [ ] Task 1 có >= 3 legal docs.
- [ ] Task 2 có >= 5 news docs.
- [ ] Task 3 có Markdown output.
- [ ] Task 4 có chunking + indexing.
- [ ] Task 5 semantic search đúng format.
- [ ] Task 6 lexical/BM25 đúng format.
- [ ] Task 7 rerank/fusion.
- [ ] Task 8 PageIndex fallback.
- [ ] Task 9 retrieve pipeline hoàn chỉnh.
- [ ] Task 10 generation có citation.

### Vòng 2 - Rà R-A-G

- [ ] R có dense search.
- [ ] R có sparse/BM25.
- [ ] R có hybrid/fusion.
- [ ] R có rerank hoặc RRF.
- [ ] A có context packaging rõ.
- [ ] A có metadata/citation id.
- [ ] A có token budget.
- [ ] A có reorder/dedup nếu cần.
- [ ] G có grounded answer.
- [ ] G có citation.
- [ ] G có abstention.

### Vòng 3 - Rà Evaluation

- [ ] Có golden dataset >= 15 Q&A.
- [ ] Có contexts retrieved cho từng question.
- [ ] Có actual answer.
- [ ] Có expected answer/ground truth.
- [ ] Có faithfulness.
- [ ] Có answer relevance.
- [ ] Có context recall.
- [ ] Có context precision.
- [ ] Có A/B test.
- [ ] Có worst performers.
- [ ] Có chẩn đoán theo RAG Triad.

### Vòng 4 - Rà Demo và Báo Cáo

- [ ] App/chatbot chạy được.
- [ ] Người dùng hỏi được câu mới.
- [ ] Answer hiển thị citation.
- [ ] Source docs hiển thị rõ.
- [ ] README nhóm có kiến trúc.
- [ ] README nhóm có phân công.
- [ ] README nhóm có hướng dẫn chạy.
- [ ] `results.md` có bảng điểm.
- [ ] Báo cáo có kết luận config tốt hơn.
- [ ] Có đề xuất cải tiến tiếp theo.

---

## 13. Lệnh Chạy Kiểm Tra

Cài dependencies:

```bash
pip install -r requirements.txt
```

Chạy test toàn bộ:

```bash
pytest tests/ -v
```

Chạy từng task:

```bash
pytest tests/test_individual.py::TestTask1 -v
pytest tests/test_individual.py::TestTask5 -v
pytest tests/test_individual.py::TestTask10 -v
```

Chạy app nhóm nếu dùng Streamlit:

```bash
streamlit run app.py
```

Chạy app nhóm nếu dùng Chainlit:

```bash
chainlit run app.py
```

Chạy eval:

```bash
python group_project/evaluation/eval_pipeline.py
```

---

## 14. Những Lỗi Dễ Bỏ Sót

- Chỉ làm dense search, quên BM25.
- Có retrieval nhưng không có rerank/fusion.
- Context không có metadata nên không citation được.
- Prompt không tách rõ system/context/question.
- Nhét quá nhiều context làm model lạc.
- Không xử lý khi thiếu evidence.
- Có chatbot nhưng không hiển thị source.
- Có eval nhưng không có golden dataset.
- Có điểm trung bình nhưng không phân tích worst performers.
- Không A/B config.
- Không chẩn đoán lỗi theo RAG Triad.
- Không ghi rõ chunk size, overlap, embedding model.
- Không có fallback khi retrieval score thấp.

---

## 15. Câu Chốt Để Demo

Day 08 không chỉ là "search rồi nhét vào prompt". Một RAG pipeline tốt cần:

```text
Hybrid Retrieval -> RRF/Rerank -> Context Augmentation sạch -> Grounded Generation có Citation -> RAGAS/DeepEval Evaluation -> A/B cải tiến
```

Khi điểm thấp, dùng RAG Triad để biết sửa ở đâu:

```text
Context Recall thấp -> sửa Retrieval/Indexing
Faithfulness thấp -> sửa Generation/Grounding
Answer Relevance thấp -> sửa Augmentation/Prompt/Query understanding
Context Precision thấp -> sửa Retrieval noise/Rerank
```
