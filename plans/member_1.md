# Thành viên #1 — Team Lead / Integration Engineer

## Vai trò
Phụ trách toàn bộ xương sống hệ thống: thiết lập repo, pipeline tích hợp, RAGAS evaluator, Regression Gate, và đảm bảo mọi module từ các thành viên khác ghép vào chạy được. Là người merge PR và unblock khi bị block.

---

## Phân công Module
| File | Nhiệm vụ |
|------|---------|
| `.github/` | Branch protection rules, PR template |
| `main.py` | `ExpertEvaluator` (RAGAS thực), Regression Gate với ngưỡng |
| `engine/runner.py` | Thêm cost tracking, latency percentiles (p50/p95), progress bar |
| `reports/` | Cấu trúc JSON output chuẩn, đảm bảo `check_lab.py` pass |

---

## Timeline 4 Giờ

### [T+0:00 — T+0:30] Phase 1: Khởi động & Thiết lập Repo
**Mục tiêu:** Toàn team có thể làm việc song song ngay lập tức.

- [ ] Tạo branch `main` protected (require PR review trước khi merge)
- [ ] Tạo 4 feature branches:
  ```
  git checkout -b feature/member1-core-pipeline
  # Tạo sẵn cho team:
  git checkout -b feature/member2-sdg-dataset
  git checkout -b feature/member3-multi-judge
  git checkout -b feature/member4-agent-retrieval
  # Push tất cả lên remote
  git push origin --all
  ```
- [ ] Tạo file `.env.example` (KHÔNG chứa key thật):
  ```
  OPENAI_API_KEY=your_openai_key_here
  ANTHROPIC_API_KEY=your_anthropic_key_here
  ```
- [ ] Kiểm tra `.gitignore` đã có: `.env`, `data/golden_set.jsonl`, `reports/`
- [ ] Tạo thư mục `reports/` và `analysis/reflections/` với `.gitkeep`
- [ ] Cập nhật `requirements.txt` — thêm các packages còn thiếu:
  ```
  anthropic>=0.20.0
  chromadb>=0.4.0
  tenacity>=8.2.0
  ```
  > Nếu dùng ragas < 0.2: thêm `datasets>=2.14.0` thay vì dùng API mới.
- [ ] **Thông báo cho team:** gửi link repo + tên branch của từng người
- [ ] **Thống nhất corpus với Member 2 & 4:** gửi 5 đoạn văn bản domain NGAY BÂY GIỜ để cả 2 dùng cùng nội dung

**Commit:** `chore: initialize repo structure, branch strategy, env template`

---

### [T+0:30 — T+1:30] Phase 2: Xây dựng Core Pipeline

#### 2A. Nâng cấp `engine/runner.py`
Thêm cost tracking và latency stats vào `run_single_test` và `run_all`:

```python
# Thêm vào run_single_test():
cost_usd = response["metadata"].get("cost_usd", 0)

return {
    "test_case": test_case["question"],
    "agent_response": response["answer"],
    "latency": latency,
    "cost_usd": cost_usd,
    "ragas": ragas_scores,
    "judge": judge_result,
    "status": "fail" if judge_result["final_score"] < 3 else "pass"
}

# Thêm vào run_all(): progress bar + aggregate stats
from tqdm.asyncio import tqdm
# wrap batch loop với tqdm
# Sau khi xong: tính p50, p95 latency từ results
```

**Commit:** `feat(runner): add cost tracking, latency percentiles, tqdm progress`

#### 2B. Implement `ExpertEvaluator` trong `main.py` với RAGAS thực

> ⚠️ **RAGAS API thay đổi lớn từ v0.2+.** Kiểm tra version trước khi code:
> ```bash
> pip show ragas  # xem version thực tế
> ```
> - **ragas < 0.2** (legacy): dùng `from ragas import evaluate` + `from datasets import Dataset`
> - **ragas >= 0.2** (current): dùng `EvaluationDataset` + `SingleTurnSample` — KHÔNG cần `datasets`

```python
# Cách viết tương thích ragas >= 0.2 (khuyên dùng):
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy
from engine.retrieval_eval import RetrievalEvaluator

_retrieval_eval = RetrievalEvaluator()

class ExpertEvaluator:
    async def score(self, case, resp):
        # RAGAS >= 0.2 API
        sample = SingleTurnSample(
            user_input=case["question"],
            response=resp["answer"],
            retrieved_contexts=resp.get("contexts", []),
            reference=case["expected_answer"]
        )
        dataset = EvaluationDataset(samples=[sample])
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        scores = result.to_pandas().iloc[0]

        # Tính Retrieval metrics từ retrieved_ids đã có trong response
        hit_rate = _retrieval_eval.calculate_hit_rate(
            expected_ids=case.get("expected_retrieval_ids", []),
            retrieved_ids=resp.get("retrieved_ids", [])
        )
        mrr = _retrieval_eval.calculate_mrr(
            expected_ids=case.get("expected_retrieval_ids", []),
            retrieved_ids=resp.get("retrieved_ids", [])
        )

        return {
            "faithfulness": float(scores.get("faithfulness", 0)),
            "relevancy": float(scores.get("answer_relevancy", 0)),
            "retrieval": {"hit_rate": hit_rate, "mrr": mrr}
        }
```

> **Nếu dùng ragas < 0.2 (legacy):** thêm `datasets>=2.14.0` vào `requirements.txt` và dùng API cũ.
> Phối hợp với Member 4: `retrieved_ids` phải có trong `resp` từ agent trước khi Member 1 code phần này.

**Commit:** `feat(evaluator): integrate ragas faithfulness and answer_relevancy`

---

### [T+1:30 — T+2:30] Phase 3: Regression Gate & Reports

#### 3A. Nâng cấp Regression Gate trong `main.py`
Thay logic delta đơn giản bằng multi-threshold gate:

```python
RELEASE_THRESHOLDS = {
    "min_avg_score":       3.5,   # LLM Judge >= 3.5/5
    "min_hit_rate":        0.70,  # Retrieval Hit Rate >= 70%
    "min_agreement_rate":  0.60,  # Judge consensus >= 60%
    "max_latency_p95":     5.0,   # P95 latency <= 5 giây
    "max_cost_per_case":   0.01   # Cost <= $0.01/case
}

def release_gate(v2_summary: dict, v1_summary: dict) -> dict:
    m = v2_summary["metrics"]
    delta = m["avg_score"] - v1_summary["metrics"]["avg_score"]
    
    checks = {
        "score_improved":    delta >= 0,
        "score_threshold":   m["avg_score"] >= RELEASE_THRESHOLDS["min_avg_score"],
        "retrieval_ok":      m["hit_rate"] >= RELEASE_THRESHOLDS["min_hit_rate"],
        "judge_consensus":   m["agreement_rate"] >= RELEASE_THRESHOLDS["min_agreement_rate"],
        "latency_ok":        m.get("p95_latency", 0) <= RELEASE_THRESHOLDS["max_latency_p95"],
        "cost_ok":           m.get("avg_cost_per_case", 0) <= RELEASE_THRESHOLDS["max_cost_per_case"],
    }
    
    decision = "APPROVE" if all(checks.values()) else "BLOCK"
    failed = [k for k, v in checks.items() if not v]
    
    return {"decision": decision, "checks": checks, "failed_checks": failed, "delta": delta}
```

**Commit:** `feat(gate): multi-threshold regression release gate with detailed checks`

#### 3B. Chuẩn hóa output `reports/summary.json`
Đảm bảo format đầy đủ để `check_lab.py` pass + có bonus fields:

```json
{
  "metadata": {
    "version": "Agent_V2_Optimized",
    "total": 50,
    "timestamp": "2026-04-21 10:00:00",
    "duration_seconds": 45.2
  },
  "metrics": {
    "avg_score": 4.1,
    "hit_rate": 0.82,
    "agreement_rate": 0.76,
    "avg_faithfulness": 0.88,
    "avg_relevancy": 0.79,
    "p50_latency": 1.2,
    "p95_latency": 3.8,
    "total_cost_usd": 0.34,
    "avg_cost_per_case": 0.0068
  },
  "regression": {
    "decision": "APPROVE",
    "delta_score": 0.3,
    "checks": {...},
    "failed_checks": []
  }
}
```

**Commit:** `feat(reports): standardize summary.json with cost, latency, regression fields`

---

### [T+2:30 — T+3:15] Phase 4: Integration & Code Review

> ⚠️ **Yêu cầu trước T+1:00:** Member 4 phải confirm interface `agent.query()` trả về đúng format
> (đặc biệt `retrieved_ids` và `metadata.cost_usd`) để Member 1 có thể code `ExpertEvaluator`.
> Nếu Member 4 chưa xong agent thực, dùng stub trả đúng format để unblock.

- [ ] Review PR của Member 2 (SDG): kiểm tra 50 cases đủ format, có `expected_retrieval_ids`
- [ ] Review PR của Member 3 (Judge): kiểm tra 2 model thật, agreement logic đúng
- [ ] Review PR của Member 4 (Agent): kiểm tra `retrieved_ids` trong response
- [ ] Merge theo thứ tự: Member4 → Member3 → Member2 → Member1
- [ ] Resolve conflicts nếu có
- [ ] Chạy integration test:
  ```bash
  python data/synthetic_gen.py
  python main.py
  python check_lab.py
  ```

**Commit:** `chore: merge all feature branches, resolve integration issues`

---

### [T+3:15 — T+3:45] Phase 5: Debug & Tối ưu

- [ ] Nếu pipeline chạy > 2 phút cho 50 cases → tăng `batch_size` hoặc tune concurrency
- [ ] Kiểm tra cost report: nếu > $0.5 tổng → đề xuất dùng `gpt-4o-mini` thay judge
- [ ] Đảm bảo `check_lab.py` ra toàn ✅ không có ⚠️
- [ ] Push tag release: `git tag v1.0.0 && git push origin v1.0.0`

---

### [T+3:45 — T+4:00] Phase 6: Hoàn thiện

- [ ] Viết `analysis/reflections/reflection_Member1.md` (xem template bên dưới)
- [ ] Tạo PR cuối cùng từ `feature/member1-core-pipeline` → `main`
- [ ] Kiểm tra lần cuối: không có file `.env` hoặc `golden_set.jsonl` trong git

---

## Git Workflow Chi tiết

```bash
# Luôn làm việc trên feature branch
git checkout feature/member1-core-pipeline

# Commit thường xuyên (mỗi 30-45 phút)
git add engine/runner.py main.py
git commit -m "feat(runner): add cost tracking and latency percentiles"

# Sync với main thường xuyên để tránh conflicts lớn
git fetch origin
git rebase origin/main

# Push và tạo PR
git push origin feature/member1-core-pipeline
gh pr create --title "feat: core pipeline with RAGAS evaluator and release gate" \
  --body "## Changes\n- RAGAS evaluator integration\n- Multi-threshold release gate\n- Cost & latency tracking"

# Review PR của team
gh pr review <PR_NUMBER> --approve
gh pr merge <PR_NUMBER> --squash
```

### Quy tắc Commit Convention
```
feat(module):    tính năng mới
fix(module):     sửa bug
chore:           setup, config, không ảnh hưởng code logic
docs:            cập nhật documentation
test:            thêm test cases
refactor:        refactor không thêm feature
```

---

## Template `analysis/reflections/reflection_Member1.md`

```markdown
# Reflection — Thành viên #1

## Module phụ trách
- engine/runner.py: async pipeline, cost tracking, latency stats
- main.py: ExpertEvaluator (RAGAS), Regression Gate
- Repo setup & integration

## Đóng góp Git
- X commits trên feature/member1-core-pipeline
- Reviewed Y PRs

## Khó khăn gặp phải
- [Mô tả vấn đề cụ thể]
- Cách giải quyết: [...]

## Kiến thức học được
- RAGAS metrics hoạt động như thế nào
- Trade-off giữa batch_size và rate limit
- Multi-threshold release gate trong CI/CD thực tế

## Đề xuất cải tiến
- [Nếu có thêm thời gian, sẽ làm gì khác?]
```

---

## Tiêu chí Hoàn thành (Definition of Done)

- [ ] `python check_lab.py` — không có lỗi ❌, không có cảnh báo ⚠️
- [ ] `reports/summary.json` có đủ: `hit_rate`, `agreement_rate`, `regression.decision`
- [ ] Pipeline chạy 50 cases trong < 2 phút
- [ ] Tất cả PR đã merge vào `main`
- [ ] Git log có commits từ cả 4 thành viên
