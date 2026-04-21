# Lab 14 — AI Evaluation Factory: Tổng quan & Kế hoạch Triển khai

## Mục tiêu Lab

Xây dựng một **Hệ thống Đánh giá Tự động (Evaluation Factory)** hoàn chỉnh để benchmark AI Agent RAG. Hệ thống phải chứng minh bằng con số cụ thể: Agent đang tốt ở đâu, tệ ở đâu, và phiên bản mới có thực sự tốt hơn không.

---

## Cấu trúc Repo

```
Lab14-AI-Evaluation-Benchmarking/
├── main.py                    # Entry point: chạy benchmark V1 vs V2, xuất reports
├── check_lab.py               # Validator định dạng bài nộp
├── requirements.txt           # pypdf, ragas, openai, python-dotenv, pandas, tqdm, jinja2
├── agent/
│   └── main_agent.py          # Agent RAG (cần thay mock bằng agent thực)
├── engine/
│   ├── runner.py              # BenchmarkRunner: async batch eval (asyncio.gather)
│   ├── llm_judge.py           # LLMJudge: multi-model scoring + agreement rate
│   └── retrieval_eval.py      # RetrievalEvaluator: Hit Rate & MRR
├── data/
│   ├── synthetic_gen.py       # SDG: tạo golden_set.jsonl via LLM
│   ├── HARD_CASES_GUIDE.md    # Hướng dẫn thiết kế adversarial/edge cases
│   └── golden_set.jsonl       # [GENERATED — không commit]
├── analysis/
│   └── failure_analysis.md    # Báo cáo 5 Whys (cần điền sau benchmark)
├── reports/                   # [GENERATED — tạo bởi main.py]
│   ├── summary.json
│   └── benchmark_results.json
└── plans/
    └── overview.md            # File này
```

---

## Trạng thái Hiện tại (Template Chưa Hoàn thiện)

| Module | File | Trạng thái | Vấn đề |
|--------|------|-----------|--------|
| Agent | `agent/main_agent.py` | Mock | Trả lời cứng, không gọi LLM/Vector DB thật |
| Judge | `engine/llm_judge.py` | Mock | Điểm cứng `score_a=4, score_b=3`, chưa gọi API |
| Retrieval Eval | `engine/retrieval_eval.py` | Partial | Logic `hit_rate`/`mrr` đúng, `evaluate_batch` hardcoded |
| SDG | `data/synthetic_gen.py` | Mock | Tạo 1 case placeholder, chưa gọi LLM |
| Runner | `engine/runner.py` | OK | Async batch logic hoạt động đúng |
| Evaluator | `main.py:ExpertEvaluator` | Mock | Điểm RAGAS cứng, chưa tính thực |
| Failure Analysis | `analysis/failure_analysis.md` | Template | Cần điền sau khi chạy |

---

## Tiêu chí Điểm Tối đa (100 điểm)

### Điểm Nhóm (60 điểm)

| Hạng mục | Điểm | Tiêu chí bắt buộc |
|----------|:----:|-------------------|
| Retrieval Evaluation | 10 | Hit Rate & MRR thực tế cho 50+ cases. Giải thích mối liên hệ Retrieval ↔ Answer Quality |
| Dataset & SDG | 10 | 50+ cases chất lượng có `expected_retrieval_ids`. Bao gồm Red Teaming cases |
| Multi-Judge Consensus | 15 | Ít nhất 2 model judge thật (GPT-4o + Claude). Agreement Rate + logic xử lý xung đột |
| Regression Testing | 10 | So sánh V1 vs V2. Logic Release Gate tự động (Approve/Block) |
| Performance (Async) | 10 | Pipeline < 2 phút cho 50 cases. Báo cáo Cost & Token usage |
| Failure Analysis | 5 | Phân tích 5 Whys sâu, chỉ ra lỗi hệ thống (Chunking, Ingestion, Retrieval, Prompting) |

> **CẢNH BÁO ĐIỂM LIỆT:** Chỉ dùng 1 Judge đơn lẻ hoặc không có Retrieval Metrics → điểm nhóm tối đa bị giới hạn ở **30/60**.

### Điểm Cá nhân (40 điểm)

| Hạng mục | Điểm | Tiêu chí |
|----------|:----:|---------|
| Engineering Contribution | 15 | Đóng góp vào Async, Multi-Judge, Metrics — chứng minh qua git commits |
| Technical Depth | 15 | Giải thích MRR, Cohen's Kappa, Position Bias, trade-off Cost vs Quality |
| Problem Solving | 10 | Cách xử lý vấn đề phát sinh khi code hệ thống phức tạp |

---

## Kế hoạch Triển khai (Theo thứ tự ưu tiên)

### Bước 1 — Thiết lập môi trường
- [ ] Tạo file `.env` với các API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- [ ] Chạy `pip install -r requirements.txt`
- [ ] Xác nhận thư mục `reports/` và `analysis/reflections/` tồn tại

### Bước 2 — Xây dựng Golden Dataset (`data/synthetic_gen.py`)
**Mục tiêu:** 50+ test cases với đủ loại độ khó

Cấu trúc mỗi case:
```json
{
  "question": "...",
  "expected_answer": "...",
  "context": "...",
  "expected_retrieval_ids": ["doc_id_1", "doc_id_2"],
  "metadata": {
    "difficulty": "easy|medium|hard|adversarial",
    "type": "fact-check|multi-hop|adversarial|out-of-context|ambiguous"
  }
}
```

Phân bổ 50 cases:
- **30 cases thông thường** (easy/medium): fact-check, multi-hop từ tài liệu domain
- **10 cases khó** (hard): câu hỏi đòi hỏi suy luận phức tạp, multi-hop 2+ bước
- **10 cases adversarial/edge** (theo `HARD_CASES_GUIDE.md`):
  - Prompt Injection / Goal Hijacking
  - Out-of-Context (Agent phải nói "không biết")
  - Ambiguous Questions
  - Conflicting Information

### Bước 3 — Kết nối Agent thực (`agent/main_agent.py`)
**Mục tiêu:** Agent trả về `retrieved_ids` để tính Retrieval Metrics

Output bắt buộc của `agent.query()`:
```python
{
  "answer": "...",
  "contexts": ["chunk text 1", "chunk text 2"],
  "retrieved_ids": ["doc_id_1", "doc_id_2"],   # BẮT BUỘC cho Retrieval Eval
  "metadata": {
    "model": "gpt-4o-mini",
    "tokens_used": 150,
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "cost_usd": 0.0003,
    "sources": ["file.pdf"]
  }
}
```

### Bước 4 — Multi-Judge Engine (`engine/llm_judge.py`)
**Mục tiêu:** 2 model judge thật + agreement logic

Rubrics chấm điểm (1-5):
- **Accuracy:** Độ chính xác so với Ground Truth
- **Faithfulness:** Chỉ dùng thông tin từ context, không hallucinate
- **Professionalism:** Ngôn ngữ chuyên nghiệp, phù hợp ngữ cảnh
- **Safety:** Không tiết lộ thông tin nhạy cảm

Logic xử lý xung đột:
```
|score_A - score_B| <= 1  → final_score = average(A, B)
|score_A - score_B| > 1   → gọi model thứ 3 làm tie-breaker
                            hoặc lấy điểm thấp hơn (conservative)
agreement_rate = 1 - (|score_A - score_B| / 4)
```

Cần implement thêm:
- **Position Bias Check:** Đổi chỗ response A↔B, so sánh điểm để detect bias
- `check_position_bias()` hiện đang `pass` — cần điền

### Bước 5 — Retrieval Evaluator (`engine/retrieval_eval.py`)
**Mục tiêu:** `evaluate_batch` tính thực từ dữ liệu agent

Logic cần implement trong `evaluate_batch`:
```python
for case in dataset:
    agent_response = await agent.query(case["question"])
    retrieved_ids = agent_response["retrieved_ids"]
    expected_ids = case["expected_retrieval_ids"]
    hit = self.calculate_hit_rate(expected_ids, retrieved_ids)
    mrr = self.calculate_mrr(expected_ids, retrieved_ids)
```

Các hàm `calculate_hit_rate` và `calculate_mrr` đã implement đúng — chỉ cần wire up `evaluate_batch`.

### Bước 6 — RAGAS Evaluator (`main.py:ExpertEvaluator`)
**Mục tiêu:** Tính faithfulness & relevancy thực bằng thư viện `ragas`

```python
from ragas.metrics import faithfulness, answer_relevancy
# Cần dataset format: question, answer, contexts, ground_truth
```

Tích hợp token tracking để báo cáo cost:
```python
"cost_per_eval_usd": total_tokens * price_per_token
```

### Bước 7 — Regression Gate (`main.py`)
**Mục tiêu:** Logic Release Gate dựa trên ngưỡng, không chỉ delta dương/âm

Ngưỡng đề xuất:
```python
THRESHOLDS = {
    "min_avg_score": 3.5,       # Điểm judge tối thiểu
    "min_hit_rate": 0.7,         # Hit rate tối thiểu 70%
    "min_agreement_rate": 0.6,   # Judge agreement tối thiểu
    "max_latency_p95": 5.0,      # Latency p95 tối đa (giây)
    "max_cost_per_case": 0.01    # Chi phí tối đa mỗi case (USD)
}
# APPROVE chỉ khi TẤT CẢ ngưỡng đều pass VÀ delta >= 0
```

### Bước 8 — Chạy Benchmark & Tạo Reports
```bash
python data/synthetic_gen.py   # Tạo golden_set.jsonl
python main.py                  # Chạy benchmark, xuất reports/
python check_lab.py             # Validate định dạng
```

Kiểm tra `reports/summary.json` có đủ các trường:
- `metadata.version`, `metadata.total`, `metadata.timestamp`
- `metrics.avg_score`, `metrics.hit_rate`, `metrics.agreement_rate`
- `metrics.avg_latency`, `metrics.total_cost_usd` (optional nhưng cộng điểm)

### Bước 9 — Failure Analysis (`analysis/failure_analysis.md`)
**Mục tiêu:** Phân tích 5 Whys sâu cho 3 case tệ nhất

Quy trình:
1. Lọc cases có `status == "fail"` hoặc `judge.final_score < 3`
2. Cluster theo loại lỗi: Hallucination, Incomplete, Tone Mismatch, Out-of-Context
3. Với 3 case tệ nhất: chạy phân tích 5 Whys chỉ ra lỗi tầng nào (Ingestion → Chunking → Retrieval → Prompting → Generation)

### Bước 10 — Individual Reflections
- Mỗi thành viên tạo file `analysis/reflections/reflection_[TenSV].md`
- Nội dung: module đảm nhận, khó khăn gặp phải, cách giải quyết, học được gì

---

## Checklist Nộp bài

```
[ ] python data/synthetic_gen.py  → data/golden_set.jsonl tồn tại, 50+ cases
[ ] python main.py                → reports/summary.json + reports/benchmark_results.json
[ ] python check_lab.py           → không có lỗi ❌
[ ] reports/summary.json có: metrics.hit_rate, metrics.agreement_rate
[ ] engine/llm_judge.py gọi 2 model thật (không phải mock)
[ ] analysis/failure_analysis.md đã điền đầy đủ (5 Whys cho 3 cases)
[ ] analysis/reflections/reflection_[TenSV].md cho từng thành viên
[ ] .env KHÔNG được commit (kiểm tra .gitignore)
[ ] Git log có commits từ nhiều thành viên (chứng minh đóng góp)
```

---

## Lưu ý Quan trọng

- **Không commit** `data/golden_set.jsonl`, `reports/`, `.env`
- **Packages còn thiếu trong `requirements.txt`** — Member 1 thêm ngay lúc setup:
  ```
  anthropic>=0.20.0    # Member 3 cần cho Claude judge
  chromadb>=0.4.0      # Member 4 cần cho Vector DB
  tenacity>=8.2.0      # Member 3 cần cho retry logic
  ```
  Nếu dùng ragas < 0.2 (legacy API): thêm `datasets>=2.14.0`.
- **RAGAS API thay đổi từ v0.2+:** dùng `EvaluationDataset` + `SingleTurnSample` thay cho `Dataset.from_dict`. Xem chi tiết trong `member_1.md`.
- **Corpus phải đồng nhất:** Member 2 viết corpus → gửi cho Member 4 → Member 4 index vào ChromaDB. `expected_retrieval_ids` trong dataset phải khớp `doc_id` trong ChromaDB.
- **Field names trong `summary.json`** phải là `hit_rate` và `agreement_rate` (không phải `avg_hit_rate`) để `check_lab.py` pass.
- Để tính cost chính xác: GPT-4o ≈ $5/1M input tokens, GPT-4o-mini ≈ $0.15/1M, Claude Haiku ≈ $0.25/1M
- Để giảm 30% cost eval: dùng fast_mode tiered judging trong `llm_judge.py` (xem `member_3.md`)

## Bonus Cộng Điểm

| Bonus | Ai làm | Mô tả |
|-------|--------|-------|
| Cohen's Kappa | Member 3 | Thêm `calculate_cohen_kappa()` vào `LLMJudge` — chính xác hơn agreement_rate |
| Correlation analysis | Member 2 | `df[["hit_rate","judge_score","faithfulness"]].corr()` trong failure analysis |
| Difficulty breakdown | Member 1/2 | Báo cáo Hit Rate & Judge Score theo từng tier easy/medium/hard/adversarial |
| Position bias report | Member 3 | Gọi `check_position_bias()` cho top-5 cases, báo cáo delta |
