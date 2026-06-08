# Bài Tập Nhóm - Search Engine / RAG Chatbot

## Mục tiêu

Nhóm xây dựng một RAG chatbot trả lời câu hỏi về pháp luật Việt Nam liên quan ma túy và các bài báo nghệ sĩ liên quan ma túy. Sản phẩm có giao diện chat, trả lời kèm citation, hiển thị source documents và có evaluation pipeline để so sánh cấu hình retrieval.

## Sản phẩm đã thực hiện

- RAG chatbot bằng Streamlit: `app.py`
- Tích hợp retrieval pipeline cá nhân: `src/task9_retrieval_pipeline.py`
- Tích hợp generation có citation: `src/task10_generation.py`
- Golden dataset 15 câu: `group_project/evaluation/golden_dataset.json`
- Evaluation script: `group_project/evaluation/eval_pipeline.py`
- Báo cáo kết quả: `group_project/evaluation/results.md`

## Kiến trúc hệ thống

```text
User
  |
  v
Streamlit Chat UI (app.py)
  |
  v
Generation with Citation (Task 10)
  |
  v
Retrieval Pipeline (Task 9)
  |
  +--> Semantic Search (Task 5)
  |
  +--> Lexical Search / BM25 (Task 6)
  |
  +--> RRF Merge / Rerank (Task 7)
  |
  +--> PageIndex / Local Fallback (Task 8)
  |
  v
Context Chunks + Source Metadata
  |
  v
Answer + Citations + Source Display
```

## Chatbot

Chatbot hỗ trợ:

- hỏi đáp nhiều lượt bằng `st.session_state`
- hiển thị câu trả lời có citation
- hiển thị source, score và nội dung evidence
- fallback extractive answer nếu chưa cấu hình `OPENAI_API_KEY`

Chạy chatbot:

```bash
streamlit run app.py
```

Nếu dùng PowerShell và môi trường ảo:

```powershell
.\.venv312\Scripts\activate
streamlit run app.py
```

## Evaluation Pipeline

Framework sử dụng: Lightweight Offline RAG Evaluator.

Lý do: chạy được local, không cần API key, vẫn đo đủ 4 metric bắt buộc:

- Faithfulness
- Answer Relevance
- Context Recall
- Context Precision

Hai cấu hình A/B:

- Config A: hybrid semantic + lexical + RRF
- Config B: lexical BM25 only

Chạy evaluation:

```bash
python group_project/evaluation/eval_pipeline.py
```

Kết quả được ghi vào:

```text
group_project/evaluation/results.md
```

Nếu thư mục `group_project/evaluation` bị khóa quyền ghi trên Windows, script sẽ ghi fallback ra file `rag_evaluation_results.generated.md` ở root repo.

## Golden Dataset

File `group_project/evaluation/golden_dataset.json` có 15 câu hỏi bao phủ:

- quy định pháp luật trong Luật Phòng, chống ma túy 2021
- Nghị định 105/2021/NĐ-CP
- Nghị định 116/2021/NĐ-CP
- các bài báo về Chi Dân, An Tây, Trúc Phương, Nguyễn Công Trí, Hữu Tín, Lê Hằng
- câu hỏi về kiến trúc RAG, fallback và citation

## Phân công công việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Lê Quang Hưng | 2A202600891 | Tích hợp retrieval pipeline Task 9 | Hoàn thành |
| Lê Văn Khoa | 2A202600603 | Golden dataset | Hoàn thành |
| Nguyễn Phúc Hiếu | 2A202600747 | evaluation | Hoàn thành |
| Nguyễn Văn Duy | 2A202600725 | Tích hợp generation/citation Task 10 | Hoàn thành |
| Trần Văn Khoa | 2A202600827 | Xây dựng Streamlit chatbot UI | Hoàn thành |
| Nghiêm Tuấn Linh | 2A2026897 |  report | Hoàn thành |

## Hướng dẫn demo

1. Cài dependencies:

```bash
pip install -r requirements.txt
```

2. Đảm bảo đã có dữ liệu markdown trong `data/standardized/`.

3. Chạy chatbot:

```bash
streamlit run app.py
```

4. Hỏi thử:

```text
Luật Phòng, chống ma túy 2021 quy định những hình thức cai nghiện nào?
```

```text
Thông tin về Chi Dân liên quan ma túy xuất hiện trong nguồn nào?
```

5. Chạy evaluation:

```bash
python group_project/evaluation/eval_pipeline.py
```

6. Mở `group_project/evaluation/results.md` để trình bày bảng điểm, worst performers và đề xuất cải tiến.

## Lưu ý kỹ thuật

- Nếu chưa có `OPENAI_API_KEY`, chatbot vẫn tạo extractive answer từ retrieved chunks để demo không bị đứng.
- Nếu thiếu `chromadb` hoặc vector store chưa build, evaluation script có local fallback để vẫn chạy A/B.
- Khi cài đủ dependencies và build vector store, chạy lại evaluation để cập nhật điểm hybrid thật.
