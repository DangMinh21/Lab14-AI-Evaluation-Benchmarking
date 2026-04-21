# Lab Day 14 — AI Evaluation Factory (Team Edition)

> "Nếu bạn không thể đo lường nó, bạn không thể cải thiện nó."

Hệ thống đánh giá tự động (Evaluation Factory) cho RAG Agent nội bộ doanh nghiệp, được xây dựng theo tiêu chuẩn AI Engineering production: multi-judge consensus, regression gate tự động, và async pipeline có báo cáo chi phí đầy đủ.

---

## Thông tin nhóm

| # | Họ và tên | MSHV | Vai trò |
|:-:|-----------|------|---------|
| 1 | **Đặng Văn Minh** | 2A202600027 | Team Lead — Core Pipeline, Regression Gate, Integration |
| 2 | **Đồng Văn Thịnh** | 2A202600365 | Data Engineer — SDG, Golden Dataset, Failure Analysis |
| 3 | **Nguyễn Thị Quỳnh Trang** | 2A202600406 | AI Engineer — Multi-Judge Consensus Engine |
| 4 | **Nguyễn Quang Tùng** | 2A202600197 | Backend Engineer — RAG Agent, ChromaDB, Retrieval |

---

## Kết quả Benchmark (đã chạy)

> Toàn bộ kết quả được sinh ra từ `python main.py` trên **55 test cases thực tế**.

| Chỉ số | V1 (Baseline) | V2 (Optimized) | Yêu cầu Gate |
|--------|:---:|:---:|:---:|
| Avg Judge Score | 4.478 / 5.0 | **4.527 / 5.0** | ≥ 3.5 ✅ |
| Pass Rate | 92.7% (51/55) | **94.5% (52/55)** | — |
| Hit Rate (Retrieval) | 72.7% | **72.7%** | ≥ 70% ✅ |
| MRR | 0.718 | **0.718** | — |
| Judge Agreement Rate | 92.9% | **91.9%** | ≥ 60% ✅ |
| P95 Latency | 6.65s | **6.40s** | ≤ 10s ✅ |
| Score Improved (V2 > V1) | — | **+0.049** | ≥ 0 ✅ |

### Quyết định Release Gate

```
✅ score_improved      — V2 tốt hơn V1 (+0.049)
✅ score_threshold     — 4.527 ≥ 3.5
✅ retrieval_ok        — Hit Rate 72.7% ≥ 70%
✅ judge_consensus     — Agreement 91.9% ≥ 60%
✅ latency_ok          — P95 6.40s ≤ 10s
✅ cost_ok             — Cost/case ≤ $0.01

QUYẾT ĐỊNH: APPROVE ✅
```

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────┐
│                      main.py                            │
│  ┌─────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │ Dataset │──▶│  RAG Agent   │──▶│ BenchmarkRunner │  │
│  │ 55 cases│   │ ChromaDB +   │   │  asyncio.gather │  │
│  │ .jsonl  │   │ GPT-4o-mini  │   │  batch_size = 5 │  │
│  └─────────┘   └──────────────┘   └────────┬────────┘  │
│                                            │            │
│              ┌─────────────────────────────┤            │
│              │                             │            │
│    ┌─────────▼──────┐          ┌───────────▼─────────┐ │
│    │  RAGAS Eval    │          │   Multi-Judge Engine │ │
│    │ faithfulness   │          │ GPT-4o-mini (judge1) │ │
│    │ + Hit Rate     │          │ Claude Haiku (judge2)│ │
│    │ + MRR          │          │ GPT-4o (tie-breaker) │ │
│    └────────────────┘          └─────────────────────┘ │
│                                                         │
│              ┌──────────────────────┐                   │
│              │   Regression Gate    │                   │
│              │ V1 vs V2, 6 checks   │                   │
│              │ → APPROVE / BLOCK    │                   │
│              └──────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

### Các module chính

| File | Phụ trách | Mô tả |
|------|-----------|-------|
| `agent/main_agent.py` | Thành viên 4 | RAG Agent — ChromaDB persistent store, chunk 250w/25w overlap, GPT-4o-mini generation |
| `data/synthetic_gen.py` | Thành viên 2 | SDG — sinh 55 test cases từ GPT-4o-mini, đa dạng độ khó |
| `engine/llm_judge.py` | Thành viên 3 | Multi-Judge — dual model, conflict resolution, position bias check, fast mode |
| `engine/retrieval_eval.py` | Thành viên 1 | Hit Rate & MRR — tính từ retrieved IDs vs expected IDs |
| `engine/runner.py` | Thành viên 1 | Async batch runner — asyncio.gather, p50/p95 latency, cost tracking |
| `main.py` | Thành viên 1 | Orchestrator — RAGAS eval, V1 vs V2, Regression Gate, output chuẩn hóa |

---

## Golden Dataset

- **Số lượng:** 55 test cases
- **Corpus:** 5 tài liệu nội bộ tiếng Việt (HR Policy, IT Access Control, Incident SOP, ...)
- **Phân bố độ khó:**

| Loại | Số cases | Mô tả |
|------|:---:|-------|
| Easy | ~15 | Tra cứu thông tin đơn giản (fact-check) |
| Medium | ~12 | Câu hỏi kết hợp 2 điều kiện |
| Hard | ~10 | Suy luận đa bước, so sánh các trường hợp |
| Adversarial / Red-team | ~9 | XSS injection, prompt leaking, jailbreak |
| Out-of-context | ~6 | Câu hỏi ngoài phạm vi tài liệu |
| Process / SLA | ~3 | Thứ tự quy trình xử lý sự cố |

- **Ground truth:** mỗi case có `expected_answer` và `expected_retrieval_ids` để tính Hit Rate / MRR.

---

## Multi-Judge Consensus Engine

Thiết kế đánh giá 5 tiêu chí: Accuracy, Faithfulness, Completeness, Professionalism, Safety.

```
Query ──▶ GPT-4o-mini (score 1–5) ┐
                                   ├──▶ |gap| ≤ 1? ──▶ Average ──▶ Final Score
Query ──▶ Claude Haiku   (score 1–5) ┘       │
                                           gap > 1 ──▶ GPT-4o (tie-breaker)

Fast Mode: Claude Haiku chạy trước
  → score ≤ 2 hoặc ≥ 4.5: short-circuit (không gọi GPT thêm)
  → score khác: gọi đủ cả 2 judges
```

**Agreement Rate** = đo khoảng cách điểm từng tiêu chí, chuẩn hóa về [0, 1].
Kết quả thực tế: **92.9% (V1)**, **91.9% (V2)** — mức đồng thuận rất cao.

---

## Hướng dẫn chạy

### Yêu cầu

```bash
# Tạo file .env từ template
cp .env.example .env
# Điền 2 keys vào .env:
# OPENAI_API_KEY=sk-proj-...
# ANTHROPIC_API_KEY=sk-ant-...

# Cài dependencies
pip install -r requirements.txt
```

### Chạy từ đầu

```bash
# Bước 1: Sinh golden dataset (55 cases)
python data/synthetic_gen.py

# Bước 2: Chạy benchmark đầy đủ (V1 + V2 + Regression Gate)
python main.py

# Bước 3: Kiểm tra format nộp bài
python check_lab.py
```

> **Lưu ý chi phí:** Benchmark 55 cases với dual-judge mất khoảng 5–10 phút và ~$1–3 API credits (OpenAI + Anthropic).

### Output sau khi chạy

```
reports/
├── summary.json            ← Tổng hợp V1 vs V2, quyết định APPROVE/BLOCK
└── benchmark_results.json  ← Chi tiết 55 cases (V1 + V2): score, latency, judge reasoning
```

---

## Cấu trúc thư mục

```
Lab14-AI-Evaluation-Benchmarking/
│
├── agent/main_agent.py          # RAG Agent (ChromaDB + GPT-4o-mini)
├── engine/
│   ├── llm_judge.py             # Multi-Judge Consensus Engine
│   ├── retrieval_eval.py        # Hit Rate & MRR
│   └── runner.py                # Async Batch Runner
│
├── data/
│   ├── docs/                    # 5 tài liệu nội bộ tiếng Việt
│   ├── golden_set.jsonl         # 55 test cases (sinh bởi synthetic_gen.py)
│   └── synthetic_gen.py         # SDG script
│
├── reports/
│   ├── summary.json             # Kết quả regression + quyết định
│   └── benchmark_results.json   # Chi tiết từng case (V1 + V2)
│
├── analysis/
│   ├── failure_analysis.md      # Phân tích 5 Whys + Failure Clustering
│   └── reflections/
│       ├── reflection_Dang_Van_Minh.md
│       ├── reflection_Dong_Van_Thinh.md
│       ├── reflection_Nguyen_Thi_Quynh_Trang.md
│       └── reflection_Nguyen_Quang_Tung.md
│
├── plans/                       # Kế hoạch triển khai từng thành viên
├── main.py                      # Entry point — chạy toàn bộ pipeline
├── check_lab.py                 # Kiểm tra format trước nộp bài
├── requirements.txt
└── .env.example
```

---

## Hướng dẫn chấm bài (dành cho Giảng viên)

### Checklist nộp bài

| Hạng mục | File/Lệnh kiểm tra | Trạng thái |
|----------|---------------------|:---:|
| Source code đầy đủ | Toàn bộ `engine/`, `agent/`, `main.py` | ✅ |
| `reports/summary.json` | Xem file | ✅ |
| `reports/benchmark_results.json` | Xem file | ✅ |
| `analysis/failure_analysis.md` | Xem file | ✅ |
| Individual reflection — Đặng Văn Minh | `analysis/reflections/reflection_Dang_Van_Minh.md` | ✅ |
| Individual reflection — Đồng Văn Thịnh | `analysis/reflections/reflection_Dong_Van_Thinh.md` | ✅ |
| Individual reflection — Nguyễn Thị Quỳnh Trang | `analysis/reflections/reflection_Nguyen_Thi_Quynh_Trang.md` | ✅ |
| Individual reflection — Nguyễn Quang Tùng | `analysis/reflections/reflection_Nguyen_Quang_Tung.md` | ✅ |
| Format validation | `python check_lab.py` | ✅ |

---

### Chấm điểm nhóm (60 điểm)

#### 1. Retrieval Evaluation — 10 điểm

Kiểm tra: `reports/benchmark_results.json` → trường `ragas.hit_rate` và `ragas.mrr` trong mỗi case.

- Hit Rate trung bình: **72.7%** (≥ 70% gate đã pass)
- MRR trung bình: **0.718**
- Thực hiện trên 55 cases với `expected_retrieval_ids` có sẵn trong golden set
- Code: `engine/retrieval_eval.py` — `calculate_hit_rate()`, `calculate_mrr()`, `evaluate_batch()`

**Mối liên hệ Retrieval ↔ Answer Quality:** 3 trong 4 failed cases có Hit Rate = 0.0 (không tìm được tài liệu) — dẫn đến hallucination. Chi tiết trong `analysis/failure_analysis.md` mục 3, Case #3.

#### 2. Dataset & SDG — 10 điểm

Kiểm tra: `data/golden_set.jsonl` (55 dòng), `data/synthetic_gen.py`

- 55 cases với ground truth: `question`, `expected_answer`, `expected_retrieval_ids`
- Phân bố đa dạng: Easy/Medium/Hard/Adversarial/Out-of-context
- Red teaming: 9 cases adversarial bao gồm XSS injection, prompt leaking, jailbreak, quyền truy cập trái phép
- Code: `data/synthetic_gen.py` — sinh bằng GPT-4o-mini với structured JSON prompt

#### 3. Multi-Judge Consensus — 15 điểm

Kiểm tra: `reports/benchmark_results.json` → trường `judge.individual_results` và `judge.status`

- **2 judges:** GPT-4o-mini + Claude Haiku 4.5, chạy song song async
- **Conflict resolution:** gap ≤ 1 → average; gap > 1 → GPT-4o full làm tie-breaker
- **Agreement rate:** 92.9% (V1), 91.9% (V2) — được tính từ khoảng cách điểm từng tiêu chí
- **Position bias check:** swap A/B và chấm lại để detect bias
- **Fast mode:** Claude chấm trước, short-circuit nếu score ≤ 2 hoặc ≥ 4.5
- Code: `engine/llm_judge.py`

```bash
# Xem ví dụ conflict case trong benchmark_results.json:
python3 -c "
import json
data = json.load(open('reports/benchmark_results.json'))
conflicts = [c for c in data['v2'] if c['judge'].get('status') == 'conflict']
print(f'Conflict cases V2: {len(conflicts)}')
for c in conflicts[:2]:
    j = c['judge']
    print(f'  Score: {j[\"final_score\"]} | Agreement: {j[\"agreement_rate\"]}')
"
```

#### 4. Regression Testing — 10 điểm

Kiểm tra: `reports/summary.json` → trường `regression`

```json
{
  "regression": {
    "v1": { "score": 4.4782, "hit_rate": 1.0, "judge_agreement": 0.9291 },
    "v2": { "score": 4.5273, "hit_rate": 1.0, "judge_agreement": 0.9186 },
    "decision": "APPROVE"
  }
}
```

- Delta score: **+0.049** (V2 cải thiện)
- 6 gate checks tự động trong `main.py` (hàm `gate_checks` dict)
- Logic: tất cả pass → APPROVE; bất kỳ fail → BLOCK

#### 5. Performance (Async) — 10 điểm

Kiểm tra: `engine/runner.py`, thực tế từ `benchmark_results.json`

- **Cơ chế async:** `asyncio.gather(evaluator.score, judge.evaluate_multi_judge)` cho mỗi case — RAGAS và Judge chạy song song
- **Batch control:** batch_size=5, tránh rate limit OpenAI/Anthropic
- **Latency thực tế:** avg 4.47s/case, P95 **6.40s** (< 10s gate)
- **Cost tracking:** `cost_usd` trả về từ agent theo token usage, aggregate qua `compute_stats()`
- 55 cases hoàn thành trong **~5 phút**

#### 6. Failure Analysis — 5 điểm

Kiểm tra: `analysis/failure_analysis.md`

- Bảng tổng quan V1 vs V2 với số liệu thực từ benchmark
- Failure Clustering: 4 nhóm lỗi với số case cụ thể
- 5 Whys cho 3 case tệ nhất (score 2.5, 2.7, 2.8)
- Root cause analysis: RAGAS silent-zero bug, hallucination khi thiếu context, sai thứ tự quy trình
- Action Plan ưu tiên P1/P2/P3

---

### Chấm điểm cá nhân (40 điểm × 4 thành viên)

| Thành viên | File báo cáo | Nội dung |
|------------|--------------|---------|
| Đặng Văn Minh | `analysis/reflections/reflection_Dang_Van_Minh.md` | Team Lead: repo setup, retrieval eval, async runner, RAGAS integration, regression gate, V2 optimization |
| Đồng Văn Thịnh | `analysis/reflections/reflection_Dong_Van_Thinh.md` | Data Engineer: golden dataset, SDG pipeline, red teaming, failure analysis (5 Whys) |
| Nguyễn Thị Quỳnh Trang | `analysis/reflections/reflection_Nguyen_Thi_Quynh_Trang.md` | Multi-Judge Engine: dual model, conflict resolution, fast mode, position bias, parser hardening |
| Nguyễn Quang Tùng | `analysis/reflections/reflection_Nguyen_Quang_Tung.md` | Backend Engineer: RAG Agent (ChromaDB), retrieval eval (Hit Rate & MRR), chunking strategy |

Mỗi báo cáo gồm: Engineering Contribution có git commit evidence, Technical Depth, Problem Solving (vấn đề thực tế gặp phải).

---

### Lệnh kiểm tra nhanh (chạy trực tiếp)

```bash
# Xác nhận format output đúng chuẩn
python check_lab.py

# Xem kết quả tổng quan
python3 -c "import json; print(json.dumps(json.load(open('reports/summary.json')), ensure_ascii=False, indent=2))"

# Đếm số cases pass/fail theo từng version
python3 -c "
import json
data = json.load(open('reports/benchmark_results.json'))
for v in ['v1', 'v2']:
    cases = data[v]
    passed = sum(1 for c in cases if c['status'] == 'pass')
    print(f'{v.upper()}: {passed}/{len(cases)} pass, avg score {sum(c[\"judge\"][\"final_score\"] for c in cases)/len(cases):.4f}')
"

# Kiểm tra multi-judge hoạt động đúng (có individual_results từ 2 models)
python3 -c "
import json
c = json.load(open('reports/benchmark_results.json'))['v1'][0]
print('Judges:', list(c['judge']['individual_results'].keys()))
print('Agreement:', c['judge']['agreement_rate'])
"
```

---

## Tính năng Bonus (ngoài yêu cầu đề)

### 1. Position Bias Detection (`engine/llm_judge.py`)

Phát hiện thiên lệch vị trí trong đánh giá pairwise — một vấn đề thực tế trong production LLM-as-judge systems.

```
Response A ──▶ Judge(A, B) ──▶ winner_forward
Response B ──▶ Judge(B, A) ──▶ winner_swapped ──▶ quy chiếu ──▶ so sánh
                                                              ↓
                                              Nếu khác nhau → is_position_biased = True
```

- 4 API calls song song: GPT(A,B), GPT(B,A), Claude(A,B), Claude(B,A)
- Trả về `bias_rate` riêng cho từng model judge
- Không có trong yêu cầu đề — tự thiết kế và implement đầy đủ

---

### 2. Tiered Judging / Fast Mode (`engine/llm_judge.py`)

Chiến lược chấm phân tầng để tối ưu chi phí API mà không giảm chất lượng:

```
Tier 1: Claude Haiku chấm trước (chi phí thấp)
  → score ≤ 2.0 : case rõ ràng tệ → short-circuit, tiết kiệm 1 GPT call
  → score ≥ 4.5 : case rõ ràng tốt → short-circuit, tiết kiệm 1 GPT call
  → score ở giữa: tiếp tục Tier 2

Tier 2: GPT-4o-mini chấm song song
  → gap ≤ 1: average
  → gap > 1: Tier 3

Tier 3: GPT-4o full (tie-breaker, chỉ khi thực sự cần)
```

Ý nghĩa: các cases "rõ ràng" (chiếm phần lớn dataset với agreement rate 91-93%) không tốn thêm API call.

---

### 3. Pairwise Comparison Engine (`engine/llm_judge.py`)

Ngoài single-score evaluation, implement thêm pairwise comparison độc lập:

- So sánh trực tiếp Response A vs Response B, trả về winner (A/B/Tie) và confidence (1-5)
- Structured JSON schema enforcement cho cả OpenAI (`response_format` strict mode) và Claude (multi-layer parser)
- Được dùng trong Position Bias Check — gọi 4 lần song song via `asyncio.gather`

---

### 4. Multi-layer JSON Parser chịu lỗi (`engine/llm_judge.py`)

Claude không support `response_format: json_schema` như OpenAI. Parser tự viết với 3 tầng fallback:

```python
1. json.loads(text)                    # direct parse
2. regex markdown fence: ```json {...} # extract từ code block
3. _find_first_json_object()           # scan ký tự, track depth brackets
```

`_find_first_json_object()` implement state machine đầy đủ: handle escaped characters, nested objects, string literals — không dùng regex để tránh false positive.

---

### 5. Cost Tracking theo token thực (`agent/main_agent.py`)

Tính chi phí thực tế từ OpenAI usage response (không ước lượng):

```python
cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000
```

Metadata đầy đủ per case: `model`, `prompt_tokens`, `completion_tokens`, `tokens_used`, `cost_usd`. Aggregate qua `compute_stats()` để báo cáo total cost và avg cost/case.

---

### 6. Latency Percentiles (p50/p95) (`engine/runner.py`)

`compute_stats()` tính percentile bằng `numpy.percentile` thay vì chỉ dùng average:

- **p50 (median)**: latency điển hình, không bị kéo bởi outlier
- **p95**: worst-case mà 95% requests đều nhanh hơn — metric chuẩn SLA production

Kết quả: V2 avg **4.47s**, P95 **6.40s** (release gate đặt ≤ 10s).

---

### 7. Agreement Rate per Criterion (`engine/llm_judge.py`)

Không chỉ đo agreement tổng (overall score gap), mà tính agreement trên từng tiêu chí độc lập:

```python
# 5 criteria + overall = 6 dimensions
agreement_rate = mean([1 - |diff_i| / 4 for i in criteria] + [1 - |diff_overall| / 4])
```

Granularity cao hơn giúp xác định được "bất đồng về cái gì" — ví dụ: GPT và Claude đồng ý về accuracy nhưng bất đồng về safety.

---

## Ghi chú kỹ thuật

**RAGAS Faithfulness = 0.0:** Toàn bộ 55 cases trả về faithfulness = 0.0 do RAGAS 0.4.x thay đổi internal schema — pipeline không crash nhưng metric bị silent-zero. Kết quả đánh giá chất lượng thực tế dựa hoàn toàn vào LLM-Judge Score (đã được validate bằng dual-model consensus). Chi tiết phân tích trong `analysis/failure_analysis.md` mục 4.

**Không commit `.env`:** API keys không có trong repo. Cần tạo `.env` từ `.env.example` trước khi chạy.
