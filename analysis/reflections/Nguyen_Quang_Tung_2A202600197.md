# Báo cáo Cá nhân — Thành viên #4 (AI/Backend Engineer / Agent & Retrieval Specialist)

## 1. Thông tin cá nhân
- Họ và tên: Nguyễn Quang Tùng
- MSSV: 2A202600197
- Vai trò trong nhóm: Xây dựng RAG Agent thực với ChromaDB Vector Store và hệ thống đánh giá Retrieval (Hit Rate & MRR).

---

## 2. Mục tiêu cá nhân trong Lab

Mục tiêu kỹ thuật của tôi trong Lab Day 14 là xây dựng tầng nền tảng cho toàn bộ pipeline:

1. Implement RAG Agent thực sự — không dùng mock — với ChromaDB persistent vector store và OpenAI embeddings.
2. Đảm bảo agent trả về `retrieved_ids` đúng format để các module khác (runner, evaluator) có thể tính Retrieval Metrics.
3. Implement `evaluate_batch()` trong `RetrievalEvaluator` để tính Hit Rate & MRR thực tế từ 55 test cases.
4. Thực nghiệm và phân tích trade-off giữa các chunking strategy (chunk_size=250 vs chunk_size=50).

---

## 3. Engineering Contribution

### 3.1 Module phụ trách

| File | Nội dung đã implement |
|------|----------------------|
| `agent/main_agent.py` | RAG Agent với ChromaDB PersistentClient, fixed-size chunking, OpenAI embeddings, GPT-4o-mini generation, cost tracking |
| `engine/retrieval_eval.py` | `evaluate_batch()` tính Hit Rate & MRR thực tế, thay thế hardcode placeholder |

### 3.2 Chi tiết triển khai `agent/main_agent.py`

**Chunking strategy:**
- Đọc trực tiếp từ `data/docs/*.txt` — không hardcode corpus
- Fixed-size chunking: `chunk_size=250 words`, `overlap=25 words` → 11 chunks / 5 docs
- Mỗi chunk lưu metadata `doc_id` để map về doc ID gốc khi eval

**Vector store:**
- ChromaDB `PersistentClient` — lưu ra `data/chromadb/`, lần sau load từ disk, không embed lại
- Embedding model: `text-embedding-3-small` ($0.02/1M tokens)
- `get_or_create_collection()` + kiểm tra `count() == 0` trước khi index

**Retrieval:**
- `collection.query()` top-3 chunks theo cosine similarity
- Map chunk IDs về doc IDs gốc (dedup, giữ thứ tự): `doc_001_chunk_2` → `doc_001`
- Trả về cả `retrieved_ids` (doc-level, cho eval) và `retrieved_chunk_ids` (chunk-level, cho failure analysis)

**Generation:**
- GPT-4o-mini với system prompt tiếng Việt, temperature=0.1
- Cost tracking thực tế: `(prompt_tokens * 0.15 + completion_tokens * 0.60) / 1_000_000`
- V1 system prompt: ngắn gọn, trả lời dựa trên tài liệu
- V2 system prompt: chi tiết hơn, yêu cầu trích dẫn chính xác, liệt kê đầy đủ điều kiện

**Output format bắt buộc:**
```python
{
    "answer": str,
    "contexts": List[str],
    "retrieved_ids": List[str],       # doc IDs — cho Hit Rate & MRR
    "retrieved_chunk_ids": List[str], # chunk IDs — cho Failure Analysis
    "metadata": {
        "model": "gpt-4o-mini",
        "tokens_used": int,
        "cost_usd": float,
        "sources": List[str],
    }
}
```

### 3.3 Chi tiết triển khai `engine/retrieval_eval.py`

Thay thế `evaluate_batch()` hardcode (`return {"avg_hit_rate": 0.85, "avg_mrr": 0.72}`) bằng tính toán thực:

```python
for case in dataset:
    retrieved_ids = case.get("retrieved_ids", [])   # inject từ runner
    expected_ids = case["expected_retrieval_ids"]   # từ golden set
    hit_rates.append(self.calculate_hit_rate(expected_ids, retrieved_ids))
    mrr_scores.append(self.calculate_mrr(expected_ids, retrieved_ids))
```

Kết quả thực tế trên 55 cases:
- **Hit Rate: 0.8909** (89%)
- **MRR: 0.8606** (86%)

Kết quả trong `reports/summary.json` sau khi tích hợp pipeline:
- **Hit Rate: 1.0** (V1 và V2 đều đạt)
- **Avg Score: 4.5873** (V1), **4.5218** (V2)

---

## 4. Technical Depth

### 4.1 Hit Rate @ K

Hit Rate @ K = 1 nếu ít nhất 1 `expected_id` nằm trong top-K `retrieved_ids`, ngược lại = 0.

```python
top_retrieved = retrieved_ids[:top_k]
return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0
```

Metric đơn giản, binary — không phân biệt được vị trí của document đúng trong top-K.

### 4.2 MRR (Mean Reciprocal Rank)

```
MRR = (1/N) * Σ (1/rank_i)
```

Trong đó `rank_i` là vị trí (1-indexed) của document đúng đầu tiên trong danh sách retrieved.

- Rank 1 → MRR = 1.0
- Rank 2 → MRR = 0.5
- Rank 3 → MRR = 0.33
- Không tìm thấy → MRR = 0.0

MRR penalize mạnh khi document đúng bị đẩy xuống thấp — phản ánh chính xác hơn Hit Rate về chất lượng ranking.

### 4.3 Mối liên hệ Retrieval ↔ Answer Quality

| Tình huống | Hệ quả |
|---|---|
| Hit Rate thấp (< 0.7) | LLM không thấy đúng context → Hallucination cao |
| Hit Rate cao nhưng MRR thấp | Document đúng có trong top-K nhưng ở rank thấp → LLM ít chú ý hơn (position bias trong attention) |
| Chunk quá lớn (250 words) | Embedding bị "loãng" → cosine similarity giữa các docs gần nhau hơn → noise lọt vào top-K |
| Chunk quá nhỏ (50 words) | Embedding tập trung → retrieval chính xác hơn, nhưng context ngắn → answer thiếu thông tin |

Thực nghiệm trong lab:
- chunk_size=250: retrieve `['doc_001', 'doc_004']` cho câu hỏi về nghỉ phép — `doc_004` (hoàn tiền) là noise vì cả hai doc đều dùng ngôn ngữ hành chính tương tự ("ngày làm việc", "quy trình", "bước 1/2/3")
- chunk_size=50: retrieve `['doc_001']` — chính xác hơn, nhưng answer "Quy trình xin nghỉ phép gồm 3 bước." thiếu chi tiết

### 4.4 Trade-off Chunk Size vs Cost

| | chunk_size=250 | chunk_size=50 |
|---|---|---|
| Số chunks | 11 | 50 |
| Retrieval precision | Thấp hơn (noise) | Cao hơn |
| Context quality | Đầy đủ hơn | Ngắn hơn |
| Cost/query | ~$0.0002 | ~$0.00008 |
| Phù hợp | Docs dài, cần context rộng | Docs ngắn, cần precision |

### 4.5 Embedding Model Choice

- `text-embedding-3-small`: $0.02/1M tokens — đủ tốt cho corpus tiếng Việt 5 docs
- `text-embedding-3-large`: tốt hơn ~20%, đắt hơn 5x — không cần thiết cho lab này

---

## 5. Problem Solving

### Vấn đề 1: Corpus không đồng bộ với golden set

**Triệu chứng:** Agent ban đầu dùng corpus hardcode (bảo mật, khiếu nại...) nhưng golden set dùng corpus thực (nghỉ phép, helpdesk, SLA...) → Hit Rate sẽ sai hoàn toàn.

**Nguyên nhân:** Không đọc `data/docs/` trực tiếp mà hardcode `DOCUMENTS = [...]`.

**Cách xử lý:** Implement `load_documents(docs_dir)` đọc tất cả file `.txt` từ `data/docs/`, dùng tên file làm doc ID. Thêm doc mới vào folder là tự động được index.

**Kết quả:** Corpus đồng bộ hoàn toàn với golden set, Hit Rate tính đúng.

### Vấn đề 2: Chunk IDs không khớp với expected_retrieval_ids

**Triệu chứng:** Sau khi implement chunking, `retrieved_ids` trả về `["doc_001_chunk_2", "doc_001_chunk_0"]` — không khớp với `expected_retrieval_ids: ["doc_001"]` trong golden set → Hit Rate = 0.

**Nguyên nhân:** ChromaDB lưu chunk IDs, không phải doc IDs.

**Cách xử lý:** Lưu `doc_id` vào metadata của mỗi chunk khi index. Khi retrieve, map chunk IDs về doc IDs gốc qua `meta["doc_id"]`, dedup và giữ thứ tự. Đồng thời giữ lại `retrieved_chunk_ids` cho Failure Analysis.

**Kết quả:** `retrieved_ids` vẫn là `["doc_001", ...]` — khớp golden set, Hit Rate tính đúng.

### Vấn đề 3: ChromaDB conflict khi re-index

**Triệu chứng:** Lần 2 khởi tạo `MainAgent()` bị lỗi `Collection already exists`.

**Nguyên nhân:** Dùng `create_collection()` thay vì `get_or_create_collection()`.

**Cách xử lý:** Đổi sang `get_or_create_collection()` + kiểm tra `collection.count() == 0` trước khi `add()`. Nếu collection đã có data thì load từ disk, không embed lại.

**Kết quả:** Agent khởi động nhanh hơn từ lần 2 trở đi, không tốn API call embedding.

### Vấn đề 4: `evaluate_batch` trả về key sai

**Triệu chứng:** `main.py` crash với `KeyError: 'hit_rate'` vì `evaluate_batch` trả về `avg_hit_rate`.

**Nguyên nhân:** Tên key không khớp với những gì `main.py` và `check_lab.py` expect.

**Cách xử lý:** Đổi key thành `hit_rate` và `mrr` theo đúng spec trong `overview.md`.

---

## 6. Minh chứng làm việc

- `agent/main_agent.py`: RAG Agent hoàn chỉnh với ChromaDB, chunking, embedding, generation, cost tracking
- `engine/retrieval_eval.py`: `evaluate_batch()` tính thực tế, không hardcode
- Test thực tế 55 cases: Hit Rate = 0.8909, MRR = 0.8606
- Pipeline `main.py` chạy thành công: `reports/summary.json` có `hit_rate: 1.0`, `avg_score: 4.5873`
- Phân tích chunking strategy: thực nghiệm chunk_size 250 vs 50, giải thích nguyên nhân noise từ embedding overlap

---

## 7. Tự đánh giá theo rubric cá nhân

### Engineering Contribution (15/15)
Đóng góp trực tiếp vào 2 module nền tảng của pipeline: RAG Agent và Retrieval Evaluator. Nếu `retrieved_ids` không có hoặc sai format, toàn bộ Retrieval Metrics (10đ nhóm) sẽ không tính được. Có thể chứng minh qua git commits trên branch `feature/member4-agent-retrieval`.

### Technical Depth (14/15)
Nắm rõ Hit Rate, MRR, trade-off chunking, embedding model choice, cost calculation. Thực nghiệm thực tế với 2 chunk sizes và giải thích được nguyên nhân kỹ thuật (embedding dilution, cosine similarity overlap). Chưa implement Cohen's Kappa (bonus).

### Problem Solving (10/10)
Xử lý 4 vấn đề thực tế phát sinh trong quá trình implement: corpus mismatch, chunk ID mapping, ChromaDB conflict, key naming mismatch. Mỗi vấn đề đều có triệu chứng, nguyên nhân và cách xử lý rõ ràng.

---

## 8. Bài học rút ra

1. **Tầng Retrieval là nền tảng** — nếu agent không trả về `retrieved_ids` đúng format, toàn bộ Retrieval Metrics sụp đổ và kéo theo 10đ nhóm.
2. **Chunking strategy ảnh hưởng cả retrieval precision lẫn answer quality** — không có giá trị tối ưu tuyệt đối, phụ thuộc vào độ dài và cấu trúc corpus.
3. **Đồng bộ interface sớm** — corpus, doc IDs, và output format của agent phải được thống nhất với các member khác trước khi code, không phải sau.
4. **Persistent vector store** tiết kiệm đáng kể API calls embedding khi chạy nhiều lần — quan trọng khi debug và test.

---

## 9. Kế hoạch nâng cấp sau Lab

1. Thử semantic chunking thay fixed-size để giảm noise từ embedding overlap.
2. Thêm reranking layer (cross-encoder) sau vector search để cải thiện MRR.
3. Cache embeddings của câu hỏi để tránh gọi API lặp lại khi chạy regression test.
4. Tính embedding cost vào tổng cost/query để báo cáo chính xác hơn.
