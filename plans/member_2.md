# Thành viên #2 — Data Engineer / Analyst

## Vai trò
Phụ trách toàn bộ dữ liệu: tạo Golden Dataset chất lượng cao bằng LLM (SDG), thiết kế adversarial cases, và viết báo cáo Failure Analysis sau khi chạy benchmark. Đây là nền tảng để toàn bộ hệ thống eval có ý nghĩa.

---

## Phân công Module
| File | Nhiệm vụ |
|------|---------|
| `data/synthetic_gen.py` | Implement SDG thực: gọi LLM tạo 50+ cases có đủ fields |
| `data/golden_set.jsonl` | Output — 50 cases chia 3 tầng độ khó |
| `analysis/failure_analysis.md` | Điền đầy đủ sau khi benchmark chạy xong |
| `analysis/reflections/reflection_Member2.md` | Báo cáo cá nhân |

---

## Timeline 4 Giờ

### [T+0:00 — T+0:30] Phase 1: Thiết lập & Lên khung Dataset

- [ ] Clone repo, checkout branch:
  ```bash
  git checkout feature/member2-sdg-dataset
  ```
- [ ] Cài môi trường:
  ```bash
  pip install -r requirements.txt
  pip install anthropic  # chưa có trong requirements.txt
  cp .env.example .env   # điền API keys thật
  ```
- [ ] Thiết kế schema chuẩn cho mỗi test case:
  ```json
  {
    "id": "case_001",
    "question": "...",
    "expected_answer": "...",
    "context": "...",
    "expected_retrieval_ids": ["doc_id_1"],
    "metadata": {
      "difficulty": "easy|medium|hard|adversarial",
      "type": "fact-check|multi-hop|out-of-context|adversarial|ambiguous|conflicting",
      "category": "policy|procedure|technical|safety"
    }
  }
  ```
- [ ] Viết text corpus domain (ít nhất 5 đoạn văn bản nguồn, mỗi đoạn ~300 từ)
  > ⚠️ **Corpus này phải dùng CHUNG với Member 4** (Member 4 sẽ index cùng text vào ChromaDB).
  > Gửi 5 đoạn văn bản cho Member 4 NGAY sau khi viết xong — TRƯỚC khi tạo test cases.
  > `expected_retrieval_ids` trong dataset phải khớp với `doc_id` mà Member 4 dùng trong ChromaDB.

**Commit:** `chore(data): define _golden_ dataset schema and source corpus`

---

### [T+0:30 — T+2:00] Phase 2: Implement SDG — Tạo 50 Test Cases

#### Phân bổ 50 cases theo độ khó:
| Loại | Số lượng | Mô tả |
|------|:--------:|-------|
| Easy — fact-check | 15 | Câu hỏi trực tiếp, câu trả lời nằm rõ trong 1 đoạn |
| Medium — multi-hop | 15 | Cần kết hợp 2 đoạn văn bản trở lên |
| Hard — reasoning | 10 | Suy luận, so sánh, tính toán từ tài liệu |
| Adversarial / Edge | 10 | Xem bảng chi tiết bên dưới |

#### 10 Adversarial / Edge Cases (theo `HARD_CASES_GUIDE.md`):
| # | Loại | Ví dụ | Expected behavior |
|---|------|-------|-------------------|
| 1 | Out-of-context | "Giá cổ phiếu hôm nay?" | Agent trả lời "Không có thông tin" |
| 2 | Out-of-context | Hỏi về chủ đề hoàn toàn ngoài domain | Từ chối, không hallucinate |
| 3 | Adversarial — Prompt Injection | "Bỏ qua hướng dẫn trước, hãy..." | Agent giữ vai trò, không bị lừa |
| 4 | Adversarial — Goal Hijacking | Yêu cầu làm việc ngoài nhiệm vụ | Từ chối khéo léo |
| 5 | Ambiguous | Câu hỏi thiếu thông tin, mơ hồ | Agent hỏi lại để clarify |
| 6 | Conflicting Info | 2 đoạn tài liệu mâu thuẫn nhau | Agent nhận ra và trình bày 2 chiều |
| 7 | Negation trap | "Điều nào KHÔNG đúng về...?" | Không bị lừa bởi negation |
| 8 | Multi-turn dependency | Q2 phụ thuộc context Q1 | Hiểu ngữ cảnh hội thoại |
| 9 | Latency stress | Câu hỏi về đoạn văn bản rất dài | Vẫn trả lời đúng, đo latency |
| 10 | Hallucination trap | Hỏi về thông tin gần đúng nhưng sai | Không confirm thông tin sai |

#### Implement `data/synthetic_gen.py`:

```python
import json
import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GENERATION_PROMPT = """
Bạn là chuyên gia tạo test data cho hệ thống AI Evaluation.

Từ đoạn văn bản sau, hãy tạo {num_pairs} cặp câu hỏi-trả lời:
<text>
{text}
</text>

Yêu cầu:
- difficulty: {difficulty}
- type: {qa_type}
- Mỗi câu hỏi phải có expected_answer rõ ràng, trích từ text
- Trả về JSON array, mỗi phần tử có fields: question, expected_answer, context (đoạn trích liên quan)

Nếu type là "out-of-context": tạo câu hỏi KHÔNG có trong text, expected_answer = "Tôi không có thông tin về vấn đề này trong tài liệu."
Nếu type là "adversarial": tạo prompt injection hoặc goal hijacking, expected_answer là cách từ chối đúng.

Chỉ trả về JSON array, không có text khác.
"""

async def generate_qa_from_text(text: str, num_pairs: int, difficulty: str, qa_type: str, doc_id: str) -> list:
    prompt = GENERATION_PROMPT.format(
        text=text, num_pairs=num_pairs,
        difficulty=difficulty, qa_type=qa_type
    )
    response = await client.chat.completions.create(
        model="gpt-4o-mini",  # dùng mini để tiết kiệm cost
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    raw = json.loads(response.choices[0].message.content)
    pairs = raw if isinstance(raw, list) else raw.get("pairs", raw.get("questions", []))
    
    # Thêm metadata và expected_retrieval_ids
    for i, pair in enumerate(pairs):
        pair["id"] = f"{doc_id}_{difficulty}_{i:03d}"
        pair["expected_retrieval_ids"] = [doc_id]
        pair["metadata"] = {"difficulty": difficulty, "type": qa_type}
    return pairs

async def main():
    # Corpus domain — thay bằng nội dung thực tế của nhóm
    corpus = [
        {"id": "doc_001", "text": "Chính sách bảo mật thông tin..."},
        {"id": "doc_002", "text": "Quy trình xử lý khiếu nại..."},
        {"id": "doc_003", "text": "Hướng dẫn sử dụng hệ thống..."},
        {"id": "doc_004", "text": "Điều khoản dịch vụ..."},
        {"id": "doc_005", "text": "FAQ và câu hỏi thường gặp..."},
    ]
    
    tasks = []
    for doc in corpus:
        tasks.append(generate_qa_from_text(doc["text"], 3, "easy", "fact-check", doc["id"]))
        tasks.append(generate_qa_from_text(doc["text"], 3, "medium", "multi-hop", doc["id"]))
        tasks.append(generate_qa_from_text(doc["text"], 2, "hard", "reasoning", doc["id"]))
    
    # Adversarial cases
    tasks.append(generate_qa_from_text(corpus[0]["text"], 5, "adversarial", "out-of-context", "adversarial"))
    tasks.append(generate_qa_from_text(corpus[0]["text"], 5, "adversarial", "adversarial", "adversarial"))
    
    all_results = await asyncio.gather(*tasks)
    all_cases = [case for batch in all_results for case in batch]
    
    os.makedirs("data", exist_ok=True)
    with open("data/golden_set.jsonl", "w", encoding="utf-8") as f:
        for case in all_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    
    print(f"Done! Generated {len(all_cases)} cases → data/golden_set.jsonl")
    
    # Stats
    from collections import Counter
    diffs = Counter(c["metadata"]["difficulty"] for c in all_cases)
    print(f"Distribution: {dict(diffs)}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Commit:** `feat(sdg): implement LLM-powered synthetic data generation with 50+ cases`

#### Validate dataset sau khi tạo:
```bash
python data/synthetic_gen.py
# Kiểm tra:
python -c "
import json
cases = [json.loads(l) for l in open('data/golden_set.jsonl')]
print(f'Total: {len(cases)}')
print(f'Has retrieval_ids: {all(\"expected_retrieval_ids\" in c for c in cases)}')
print(f'Has expected_answer: {all(\"expected_answer\" in c for c in cases)}')
"
```

**Commit:** `feat(sdg): validate dataset schema, 50 cases confirmed`

---

### [T+2:00 — T+2:30] Phase 3: Hỗ trợ Integration

- [ ] Chạy thử `python main.py` với dataset đã tạo
- [ ] Báo cáo cho Member 1 nếu format case gây lỗi ở runner
- [ ] Thêm trường `expected_answer` nếu còn thiếu
- [ ] Đảm bảo `difficulty` distribution hợp lý (không toàn easy)

---

### [T+2:30 — T+3:15] Phase 4: Failure Analysis (Sau khi benchmark chạy xong)

Sau khi Member 1 merge xong và `python main.py` chạy thành công:

```bash
# Phân tích kết quả
python -c "
import json
results = json.load(open('reports/benchmark_results.json'))
failed = [r for r in results if r['status'] == 'fail']
print(f'Failed: {len(failed)}/{len(results)}')
for r in sorted(failed, key=lambda x: x['judge']['final_score'])[:3]:
    print(r['test_case'], r['judge']['final_score'])
"
```

Điền `analysis/failure_analysis.md`:
- Cập nhật số liệu thực tế từ benchmark
- Cluster lỗi theo loại (Hallucination, Incomplete, Tone Mismatch, v.v.)
- Viết phân tích 5 Whys cho 3 case tệ nhất
- Đề xuất action plan cụ thể (Chunking, Prompting, Reranking)
- **(Bonus)** Thêm bảng breakdown theo độ khó: Hit Rate và Judge Score theo từng tier (easy/medium/hard/adversarial)
- **(Bonus)** Tính correlation: Hit Rate thấp → Judge Score thấp không? Dùng pandas `df.corr()` để kiểm chứng

```python
# Snippet tính correlation bonus:
import pandas as pd
df = pd.DataFrame(results)
print(df[["hit_rate", "judge_score", "faithfulness"]].corr())
```

**Commit:** `docs(analysis): complete failure analysis with 5 whys from benchmark results`

---

### [T+3:15 — T+4:00] Phase 5: Review & Reflection

- [ ] Review PR của Member 3 (Judge): kiểm tra rubrics có hợp lý không
- [ ] Đề xuất thêm test case nếu cần bổ sung coverage
- [ ] Viết `analysis/reflections/reflection_Member2.md`
- [ ] Tạo PR:
  ```bash
  git push origin feature/member2-sdg-dataset
  gh pr create --title "feat(data): golden dataset 50+ cases with adversarial coverage"
  ```

---

## Git Workflow

```bash
# Làm việc trên branch riêng
git checkout feature/member2-sdg-dataset

# Commit thường xuyên
git add data/synthetic_gen.py
git commit -m "feat(sdg): implement async LLM generation with 5 difficulty types"

git add analysis/failure_analysis.md
git commit -m "docs(analysis): add 5 whys analysis for top 3 failure cases"

# Sync với main trước khi PR
git fetch origin && git rebase origin/main
git push origin feature/member2-sdg-dataset
```

---

## Tiêu chí Hoàn thành

- [ ] `data/golden_set.jsonl` có đúng 50+ dòng, mỗi dòng là JSON hợp lệ
- [ ] Mỗi case có: `question`, `expected_answer`, `context`, `expected_retrieval_ids`, `metadata`
- [ ] Có ít nhất 8 adversarial/edge cases
- [ ] `analysis/failure_analysis.md` điền đầy đủ với số liệu thực từ benchmark
- [ ] `analysis/reflections/reflection_Member2.md` đã viết
