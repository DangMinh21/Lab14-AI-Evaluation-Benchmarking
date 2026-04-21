# Thành viên #3 — AI Engineer / Multi-Judge Specialist

## Vai trò
Phụ trách toàn bộ hệ thống chấm điểm: implement Multi-Judge Engine với 2 model thật (GPT-4o + Claude Haiku), tính Agreement Rate, xử lý xung đột tự động, và detect Position Bias. Đây là module quan trọng nhất — thiếu multi-judge sẽ bị giới hạn điểm liệt.

---

## Phân công Module
| File | Nhiệm vụ |
|------|---------|
| `engine/llm_judge.py` | Multi-Judge Engine hoàn chỉnh |
| `requirements.txt` | Thêm `anthropic` package |
| `analysis/reflections/reflection_Member3.md` | Báo cáo cá nhân |

---

## Timeline 4 Giờ

### [T+0:00 — T+0:30] Phase 1: Thiết lập

- [ ] Clone repo, checkout branch:
  ```bash
  git checkout feature/member3-multi-judge
  ```
- [ ] Cài packages:
  ```bash
  pip install openai anthropic python-dotenv
  cp .env.example .env   # điền API keys thật
  ```
- [ ] Test kết nối API nhanh:
  ```python
  from openai import AsyncOpenAI
  import anthropic
  # Test gọi nhanh 1 message mỗi API để xác nhận key hợp lệ
  ```
- [ ] Thêm các packages còn thiếu vào `requirements.txt`:
  ```
  anthropic>=0.20.0
  tenacity>=8.2.0
  ```

**Commit:** `chore(deps): add anthropic and tenacity to requirements`

---

### [T+0:30 — T+2:30] Phase 2: Implement Multi-Judge Engine

#### Kiến trúc tổng thể `engine/llm_judge.py`:

```
Input: (question, answer, ground_truth)
   │
   ├──► Judge A: GPT-4o-mini  ──► score_a (1-5) + reasoning_a
   ├──► Judge B: Claude Haiku ──► score_b (1-5) + reasoning_b
   │
   ▼
Agreement Check: |score_a - score_b|
   ├── <= 1: final = average, agreement = "high"
   └── >  1: Tie-breaker (GPT-4o full) ──► final = tie_score, agreement = "low"
   │
   ▼
Position Bias Check (optional, nếu còn thời gian):
   Swap answer positions → re-score → compare
   │
   ▼
Output: { final_score, agreement_rate, individual_scores, reasoning, conflict_resolved }
```

#### Implement đầy đủ:

```python
import asyncio
import os
from typing import Dict, Any
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

RUBRIC_PROMPT = """
Bạn là một Evaluator AI chuyên nghiệp. Hãy chấm điểm câu trả lời sau theo thang điểm 1-5.

**Câu hỏi:** {question}
**Ground Truth (Câu trả lời chuẩn):** {ground_truth}
**Câu trả lời cần đánh giá:** {answer}

**Tiêu chí chấm (mỗi tiêu chí 0-1 điểm, tổng quy đổi về 1-5):**
1. Accuracy (độ chính xác so với Ground Truth)
2. Faithfulness (chỉ dùng thông tin từ context, không hallucinate)
3. Completeness (trả lời đủ ý, không bỏ sót)
4. Professionalism (ngôn ngữ chuyên nghiệp, phù hợp)
5. Safety (không tiết lộ thông tin nhạy cảm, không gây hại)

**Thang điểm:**
- 5: Hoàn hảo, đáp ứng tất cả tiêu chí
- 4: Tốt, có vài điểm nhỏ cần cải thiện
- 3: Trung bình, đúng nhưng thiếu hoặc có sai sót nhỏ
- 2: Kém, sai hoặc thiếu nhiều
- 1: Rất kém, sai hoàn toàn hoặc hallucinate nặng

Trả về JSON: {{"score": <int 1-5>, "reasoning": "<giải thích ngắn gọn 1-2 câu>", "criteria": {{"accuracy": <0-1>, "faithfulness": <0-1>, "completeness": <0-1>, "professionalism": <0-1>, "safety": <0-1>}}}}
"""

class LLMJudge:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.conflict_threshold = 1  # điểm lệch > 1 → cần tie-breaker

    async def _judge_with_gpt(self, question: str, answer: str, ground_truth: str, model: str = "gpt-4o-mini") -> Dict:
        prompt = RUBRIC_PROMPT.format(question=question, answer=answer, ground_truth=ground_truth)
        resp = await self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        import json
        result = json.loads(resp.choices[0].message.content)
        return {
            "score": result["score"],
            "reasoning": result.get("reasoning", ""),
            "criteria": result.get("criteria", {}),
            "model": model,
            "tokens": resp.usage.total_tokens
        }

    async def _judge_with_claude(self, question: str, answer: str, ground_truth: str) -> Dict:
        prompt = RUBRIC_PROMPT.format(question=question, answer=answer, ground_truth=ground_truth)
        resp = await self.anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        import json
        # Claude không có response_format, cần extract JSON từ text
        content = resp.content[0].text
        # Tìm JSON trong response
        start = content.find("{")
        end = content.rfind("}") + 1
        result = json.loads(content[start:end])
        return {
            "score": result["score"],
            "reasoning": result.get("reasoning", ""),
            "criteria": result.get("criteria", {}),
            "model": "claude-haiku-4-5",
            "tokens": resp.usage.input_tokens + resp.usage.output_tokens
        }

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        # Gọi 2 judge song song
        judge_a_task = self._judge_with_gpt(question, answer, ground_truth, model="gpt-4o-mini")
        judge_b_task = self._judge_with_claude(question, answer, ground_truth)
        
        result_a, result_b = await asyncio.gather(judge_a_task, judge_b_task)
        
        score_a = result_a["score"]
        score_b = result_b["score"]
        diff = abs(score_a - score_b)
        conflict_resolved = False
        
        if diff > self.conflict_threshold:
            # Tie-breaker: dùng GPT-4o full (đắt hơn nhưng chính xác hơn)
            tie_result = await self._judge_with_gpt(question, answer, ground_truth, model="gpt-4o")
            final_score = tie_result["score"]
            conflict_resolved = True
            total_tokens = result_a["tokens"] + result_b["tokens"] + tie_result["tokens"]
        else:
            final_score = round((score_a + score_b) / 2, 1)
            total_tokens = result_a["tokens"] + result_b["tokens"]
        
        # Agreement rate: 1.0 nếu đồng ý hoàn toàn, giảm dần theo mức lệch
        agreement_rate = max(0.0, 1.0 - (diff / 4.0))
        
        return {
            "final_score": final_score,
            "agreement_rate": round(agreement_rate, 2),
            "conflict_resolved": conflict_resolved,
            "individual_scores": {
                "gpt-4o-mini": score_a,
                "claude-haiku": score_b
            },
            "reasoning": f"GPT: {result_a['reasoning']} | Claude: {result_b['reasoning']}",
            "criteria_avg": self._avg_criteria(result_a.get("criteria", {}), result_b.get("criteria", {})),
            "total_tokens": total_tokens,
            "cost_usd": self._estimate_cost(result_a, result_b, conflict_resolved)
        }

    def _avg_criteria(self, criteria_a: dict, criteria_b: dict) -> dict:
        keys = set(criteria_a.keys()) | set(criteria_b.keys())
        return {k: round((criteria_a.get(k, 0) + criteria_b.get(k, 0)) / 2, 2) for k in keys}

    def calculate_cohen_kappa(self, scores_a: list, scores_b: list) -> float:
        """
        (Bonus) Cohen's Kappa — chính xác hơn agreement_rate đơn giản vì điều chỉnh
        cho yếu tố ngẫu nhiên. Gọi sau khi chạy toàn bộ dataset.
        Trả về kappa trong [-1, 1]: >= 0.6 là agreement tốt.
        """
        from collections import Counter
        n = len(scores_a)
        if n == 0:
            return 0.0
        # Observed agreement
        p_o = sum(1 for a, b in zip(scores_a, scores_b) if a == b) / n
        # Expected agreement (tích xác suất từng nhãn)
        count_a = Counter(scores_a)
        count_b = Counter(scores_b)
        labels = set(count_a.keys()) | set(count_b.keys())
        p_e = sum((count_a.get(k, 0) / n) * (count_b.get(k, 0) / n) for k in labels)
        if p_e == 1.0:
            return 1.0
        return round((p_o - p_e) / (1 - p_e), 4)

    def _estimate_cost(self, result_a: dict, result_b: dict, conflict_resolved: bool) -> float:
        # GPT-4o-mini: $0.15/1M input, $0.60/1M output (simplified: $0.15/1M avg)
        # Claude Haiku: $0.25/1M input, $1.25/1M output (simplified: $0.25/1M avg)
        # GPT-4o (tie-breaker): $5/1M input, $15/1M output (simplified: $5/1M avg)
        cost = result_a["tokens"] * 0.15 / 1_000_000
        cost += result_b["tokens"] * 0.25 / 1_000_000
        if conflict_resolved:
            cost += 500 * 5.0 / 1_000_000  # ước tính tie-breaker ~500 tokens
        return round(cost, 6)

    async def check_position_bias(self, question: str, answer_a: str, answer_b: str, ground_truth: str) -> Dict:
        """
        Detect position bias: chấm answer_a hai lần trong prompt so sánh 2 responses.
        Lần 1: answer_a ở vị trí 1 (đứng trước). Lần 2: answer_a ở vị trí 2 (đứng sau).
        Nếu điểm thay đổi → judge bị ảnh hưởng bởi vị trí → có position bias.
        """
        COMPARISON_PROMPT = (
            "Câu hỏi: {question}\n"
            "Ground Truth: {ground_truth}\n\n"
            "Có 2 câu trả lời. Hãy chấm điểm câu trả lời được đánh dấu [TARGET] (thang 1-5):\n\n"
            "Response 1: {response_1}\n"
            "Response 2: {response_2}\n\n"
            "[TARGET] là Response {target_position}.\n\n"
            'Trả về JSON: {{"score": <int 1-5>, "reasoning": "<giải thích>"}}'
        )

        prompt_a_first = COMPARISON_PROMPT.format(
            question=question, ground_truth=ground_truth,
            response_1=answer_a, response_2=answer_b, target_position=1
        )
        prompt_a_second = COMPARISON_PROMPT.format(
            question=question, ground_truth=ground_truth,
            response_1=answer_b, response_2=answer_a, target_position=2
        )

        import json
        resp1, resp2 = await asyncio.gather(
            self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_a_first}],
                response_format={"type": "json_object"},
                temperature=0
            ),
            self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_a_second}],
                response_format={"type": "json_object"},
                temperature=0
            )
        )

        score_when_first = json.loads(resp1.choices[0].message.content)["score"]
        score_when_second = json.loads(resp2.choices[0].message.content)["score"]
        delta = abs(score_when_first - score_when_second)

        return {
            "bias_detected": delta > 0,
            "score_when_first": score_when_first,
            "score_when_second": score_when_second,
            "delta": delta
        }
```

**Commit:** `feat(judge): implement multi-judge with GPT-4o-mini and Claude Haiku`

---

### [T+2:00 — T+2:30] Phase 3: Unit Test & Validate

Viết test nhanh để xác nhận judge hoạt động:

```python
# Chạy: python -m engine.llm_judge
if __name__ == "__main__":
    async def test():
        judge = LLMJudge()
        result = await judge.evaluate_multi_judge(
            question="Chính sách bảo mật dữ liệu của công ty là gì?",
            answer="Công ty cam kết bảo vệ dữ liệu người dùng theo GDPR.",
            ground_truth="Công ty áp dụng mã hóa AES-256 và tuân thủ GDPR và ISO 27001."
        )
        print(f"Score: {result['final_score']}")
        print(f"Agreement: {result['agreement_rate']}")
        print(f"Conflict resolved: {result['conflict_resolved']}")
        print(f"Cost: ${result['cost_usd']}")
    
    asyncio.run(test())
```

Kiểm tra:
- [ ] Không throw exception
- [ ] `final_score` là float trong [1, 5]
- [ ] `agreement_rate` trong [0, 1]
- [ ] `individual_scores` có cả `gpt-4o-mini` và `claude-haiku`

**Commit:** `test(judge): add validation test for multi-judge output format`

---

### [T+2:30 — T+3:00] Phase 4: Tối ưu Cost & Tích hợp

#### Chiến lược giảm 30% cost eval (bonus điểm):
```
Chiến lược "tiered judging":
1. Chạy Claude Haiku trước (rẻ nhất)
2. Nếu score < 2 hoặc score >= 4.5 → chắc chắn, không cần judge thứ 2
3. Chỉ chạy GPT-4o-mini khi score ở vùng ambiguous [2, 4.5]
4. Tie-breaker GPT-4o chỉ khi conflict > 1 điểm

Tiết kiệm ước tính: ~30% cases không cần full dual-judge
```

Implement flag `fast_mode`:
```python
async def evaluate_multi_judge(self, ..., fast_mode: bool = False):
    if fast_mode:
        # Chạy Claude Haiku trước
        quick_result = await self._judge_with_claude(...)
        if quick_result["score"] <= 2 or quick_result["score"] >= 4.5:
            return {  # Chắc chắn, không cần judge thứ 2
                "final_score": quick_result["score"],
                "agreement_rate": 1.0,
                "individual_scores": {"claude-haiku": quick_result["score"]},
                ...
            }
    # Tiếp tục dual-judge nếu không fast_mode hoặc score ở vùng ambiguous
```

**Commit:** `feat(judge): add fast_mode tiered judging to reduce cost by ~30%`

---

### [T+3:00 — T+3:45] Phase 5: Integration Support

- [ ] Xác nhận interface của `evaluate_multi_judge` khớp với `engine/runner.py`
- [ ] Kiểm tra output có đủ fields mà `main.py` cần: `final_score`, `agreement_rate`
- [ ] Hỗ trợ Member 1 nếu có lỗi khi integrate judge vào pipeline
- [ ] Đảm bảo error handling khi API timeout hoặc rate limit:
  ```python
  import asyncio
  from tenacity import retry, stop_after_attempt, wait_exponential
  
  @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
  async def _judge_with_gpt(self, ...):
      ...
  ```

**Commit:** `feat(judge): add retry logic for API rate limits and timeouts`

---

### [T+3:45 — T+4:00] Phase 6: PR & Reflection

- [ ] Tạo PR:
  ```bash
  git push origin feature/member3-multi-judge
  gh pr create --title "feat(judge): multi-judge engine with GPT-4o-mini + Claude Haiku" \
    --body "## Changes
  - Dual model judging (GPT-4o-mini + Claude Haiku-4-5)
  - Agreement rate calculation
  - Conflict resolution via GPT-4o tie-breaker
  - Position bias detection
  - Fast mode for 30% cost reduction
  - Retry logic for rate limits"
  ```
- [ ] Viết `analysis/reflections/reflection_Member3.md`

---

## Git Workflow

```bash
git checkout feature/member3-multi-judge

# Commit theo từng milestone
git commit -m "feat(judge): implement GPT-4o-mini judge with rubric scoring"
git commit -m "feat(judge): add Claude Haiku judge and dual consensus logic"
git commit -m "feat(judge): conflict resolution and agreement rate calculation"
git commit -m "feat(judge): position bias check and fast mode cost optimization"
git commit -m "fix(judge): retry logic for API errors"

# Sync và push
git fetch origin && git rebase origin/main
git push origin feature/member3-multi-judge
```

---

## Tiêu chí Hoàn thành

- [ ] `evaluate_multi_judge()` gọi 2 model thật (không mock)
- [ ] `individual_scores` có cả 2 key model trong output
- [ ] `agreement_rate` tính từ diff thực tế, không hardcode
- [ ] Logic conflict resolution hoạt động khi diff > 1
- [ ] `cost_usd` được ước tính và trả về trong mỗi lần eval
- [ ] Không throw exception khi API trả lỗi (có retry)
- [ ] `anthropic` đã được thêm vào `requirements.txt`

---

## Kiến thức Cần Nắm (cho phần Điểm Cá nhân)

**Agreement Rate vs Cohen's Kappa:**
- Agreement Rate đơn giản: `1 - |diff|/4` — dễ tính, dễ hiểu
- Cohen's Kappa: điều chỉnh cho yếu tố ngẫu nhiên — chính xác hơn trong nghiên cứu
- Trong sản phẩm thực tế: Agreement Rate thường đủ dùng

**Position Bias:**
- LLM judge thường ưu tiên response xuất hiện đầu tiên
- Cách detect: đổi thứ tự A↔B, nếu điểm thay đổi → có bias
- Cách fix: average score của 2 lần đổi thứ tự

**Trade-off Cost vs Quality:**
- GPT-4o: chính xác nhất, đắt nhất (~$5/1M tokens)
- GPT-4o-mini: 97% chính xác so với GPT-4o, rẻ hơn 33x
- Claude Haiku-4-5: nhanh nhất, rẻ, phù hợp làm judge thứ 2
- Chiến lược tối ưu: dùng cheap model làm pre-filter, chỉ escalate khi cần
