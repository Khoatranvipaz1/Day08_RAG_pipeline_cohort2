# Báo Cáo Cá Nhân Day 08 - RAG Pipeline

## Thông Tin Sinh Viên

- Họ và tên: Trần Văn Khoa
- MSV: `2A202600827`
- Branch: `TVKhoa`

## Phạm Vi Thực Hiện

Đã triển khai pipeline:

```text
Documents -> Standardization -> Chunking -> Indexing
Question -> Hybrid Retrieval -> RRF/Rerank
         -> Context Augmentation -> Grounded Generation -> Citation
```

## Cấu Hình Kỹ Thuật

### Chunking

- Loại: fixed-size character chunking có overlap.
- `CHUNK_SIZE = 900`.
- `CHUNK_OVERLAP = 120`.
- Implementation: `split_text()` trong `src/rag_utils.py`.

### Embedding và Index

- Local mặc định: `scikit-learn` `TfidfVectorizer`.
- OpenAI embedding: `text-embedding-3-small` khi quota khả dụng.
- Jina fallback: `jina-embeddings-v3`.
- Vector database optional: Weaviate class `Day08Chunk`.

### Retrieval

- Semantic retrieval: TF-IDF cosine hoặc Weaviate vector search.
- Lexical retrieval: BM25 bằng `rank-bm25`.
- Fusion: Reciprocal Rank Fusion.
- Reranking: Jina Reranker, fallback term overlap local.
- Fallback: local vectorless keyword search.

### Generation

- Grounded system prompt chỉ cho phép trả lời từ context.
- Có citation dạng `[doc_id]`.
- Có hỏi lại khi câu hỏi quá mơ hồ.
- Có abstention khi thiếu evidence.
- Không suy đoán danh tính hoặc dữ liệu cá nhân.

## Kết Quả

- Pytest cá nhân: `33 passed, 2 skipped, 0 failed`.
- Hai test skipped là optional integration checks cho PageIndex thật và generation API.
- Data đã chuẩn hóa vào `data/standardized/`.
- Local ingestion đã sinh `data/index/chunks.json`.
- Weaviate đã được smoke test ingestion và semantic search.
- Chatbot Streamlit chạy tại `group_project/app.py`.
- Evaluation nhóm có golden dataset 15 câu và A/B dense-only với hybrid.

## Lưu Ý

Một số tài liệu trong corpus là dữ liệu mẫu phục vụ lab, không thay thế văn bản
pháp luật hoặc bài báo chính thức. Hệ thống hiển thị source và từ chối xác minh
khi context không đủ.
