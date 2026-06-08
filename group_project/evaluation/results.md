# RAG Evaluation Results

## Framework sử dụng

Framework: Lightweight Offline RAG Evaluator.

Lý do chọn: chạy được local, không cần API key, vẫn bám theo 4 trục đánh giá của DeepEval/RAGAS: faithfulness, answer relevance, context recall, context precision.

Ghi chú lần chạy hiện tại: Python hệ thống đang thiếu `chromadb` và `numpy`, nên Config A fallback về local lexical retrieval khi semantic index chưa sẵn sàng. Khi cài đủ dependencies và build vector store, chạy lại `eval_pipeline.py` để cập nhật điểm hybrid thật.

## Overall Scores

| Metric | Config A (hybrid + RRF) | Config B (BM25 only) | Delta |
|--------|--------------------------|----------------------|-------|
| Faithfulness | 0.853 | 0.853 | +0.000 |
| Answer Relevance | 0.681 | 0.681 | +0.000 |
| Context Recall | 0.500 | 0.500 | +0.000 |
| Context Precision | 1.000 | 1.000 | +0.000 |
| Average | 0.758 | 0.758 | +0.000 |

## A/B Comparison Analysis

**Config A:** chạy retrieval pipeline nhóm bằng semantic search + lexical search, fusion bằng RRF và fallback nếu cần.

**Config B:** chỉ dùng lexical BM25 trên corpus markdown/chunks.

**Kết luận:** Config A và Config B đang ngang điểm trong lần chạy hiện tại; điều này thường xảy ra khi semantic index hoặc dependency ML chưa sẵn sàng và Config A phải fallback về lexical/local retrieval. Nếu Config A thấp hoặc chỉ ngang Config B, nguyên nhân thường là semantic index chưa đầy đủ, embedding chưa tải được, hoặc dữ liệu markdown pháp luật convert còn nhiễu.

## Worst Performers (Bottom 3 - Config A)

| # | Question | Faithfulness | Relevance | Recall | Precision | Top Sources |
|---|----------|--------------|-----------|--------|-----------|-------------|
| 1 | Fallback PageIndex hoặc vectorless fallback dùng khi nào? | 0.720 | 0.250 | 0.000 | 1.000 | luat_73_2021_qh14_phong_chong_ma_tuy.md |
| 2 | Nghị định 116/2021/NĐ-CP liên quan đến nội dung nào trong lĩnh vực cai nghiện ma túy? | 0.611 | 0.500 | 0.000 | 1.000 | luat_73_2021_qh14_phong_chong_ma_tuy.md, article_01_congan_tphcm_chi_dan_an_tay_truc_phuong.md |
| 3 | Vì sao generation cần citation và cơ chế không suy đoán? | 0.837 | 0.333 | 0.000 | 1.000 | luat_73_2021_qh14_phong_chong_ma_tuy.md |

## Recommendations

### Cải tiến 1
**Action:** Làm sạch markdown pháp luật sau khi convert, đặc biệt các phần header/footer và ký tự nhiễu từ PDF.  
**Expected impact:** Tăng context precision và giúp citation chính xác hơn.

### Cải tiến 2
**Action:** Bổ sung golden dataset với câu hỏi có expected_context trỏ tới đúng điều/khoản hoặc đúng bài báo.  
**Expected impact:** Đánh giá context recall sát thực tế hơn.

### Cải tiến 3
**Action:** Nếu có API key, bật reranker cross-encoder hoặc LLM generation thật trong Task 10.  
**Expected impact:** Tăng answer relevance và chất lượng câu trả lời cuối.
