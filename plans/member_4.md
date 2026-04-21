# Thành viên #4 — AI/Backend Engineer / Agent & Retrieval Specialist

## Vai trò
Phụ trách Agent RAG thực và hệ thống đánh giá Retrieval. Đây là tầng nền tảng — nếu Agent không trả về `retrieved_ids` thì toàn bộ Retrieval Metrics sẽ không tính được, ảnh hưởng trực tiếp đến 10 điểm Retrieval Evaluation. Cần hoàn thành sớm để các module khác có thể test.

---

## Phân công Module
| File | Nhiệm vụ |
|------|---------|
| `agent/main_agent.py` | RAG Agent thực với Vector DB + LLM, trả về `retrieved_ids` |
| `engine/retrieval_eval.py` | Wire up `evaluate_batch` với agent thực |
| `analysis/reflections/reflection_Member4.md` | Báo cáo cá nhân |

---

## Timeline 4 Giờ

### [T+0:00 — T+0:30] Phase 1: Thiết lập & Quyết định Kiến trúc

- [ ] Clone repo, checkout branch:
  ```bash
  git checkout feature/member4-agent-retrieval
  ```
- [ ] Cài packages cần thiết:
  ```bash
  pip install -r requirements.txt
  pip install chromadb openai python-dotenv  # hoặc faiss-cpu tùy chọn
  cp .env.example .env
  ```
- [ ] **⚠️ QUAN TRỌNG — Đồng bộ corpus với Member 2 TRƯỚC KHI CODE:**
  Member 2 tạo `golden_set.jsonl` với `expected_retrieval_ids: ["doc_001"...]`.
  Member 4 index ChromaDB với cùng `doc_001...doc_005`.
  **Nếu text corpus khác nhau → Hit Rate sẽ sai hoàn toàn.** Cần thống nhất cùng 5 đoạn văn bản ngay lúc setup.
- [ ] **Confirm agent interface với Member 1 trước T+1:00:**
  ```python
  # Interface bắt buộc — KHÔNG thay đổi tên field:
  {
      "answer": str,
      "contexts": List[str],
      "retrieved_ids": List[str],   # BẮT BUỘC
      "metadata": {
          "model": str,
          "cost_usd": float,        # BẮT BUỘC cho cost tracking
          "tokens_used": int,
      }
  }
  ```
- [ ] **Quyết định kiến trúc Vector DB** (chọn 1):
  - **Option A — ChromaDB** (khuyên dùng): đơn giản, in-memory, không cần server
  - **Option B — FAISS**: nhanh hơn, phù hợp dataset lớn
  - **Option C — Mock với ID mapping**: nếu không có Vector DB thực, dùng dict lookup theo `doc_id`

> Nếu thời gian hạn chế → dùng **Option C** (mock retrieval với ID mapping đúng) vẫn đủ để tính Hit Rate & MRR thực tế, miễn là agent trả về `retrieved_ids` đúng format.

**Commit:** `chore(agent): setup dependencies and decide on retrieval architecture`

---

### [T+0:30 — T+2:00] Phase 2: Implement Agent RAG

#### Option A — ChromaDB (Full implementation):

```python
# agent/main_agent.py
import asyncio
import os
import json
from typing import Dict, List
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class MainAgent:
    """RAG Agent sử dụng ChromaDB + OpenAI"""
    
    def __init__(self):
        self.name = "SupportAgent-v1"
        self.llm = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._init_vector_store()

    def _init_vector_store(self):
        """Khởi tạo ChromaDB in-memory và index documents"""
        import chromadb
        from chromadb.utils import embedding_functions
        
        self.chroma_client = chromadb.Client()
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        self.collection = self.chroma_client.create_collection(
            name="knowledge_base",
            embedding_function=openai_ef
        )
        self._load_documents()

    def _load_documents(self):
        """
        Load và index documents từ corpus.
        Thay thế danh sách này bằng tài liệu domain thực tế của nhóm.
        """
        documents = [
            {"id": "doc_001", "text": "Chính sách bảo mật thông tin: Công ty áp dụng mã hóa AES-256 và tuân thủ GDPR và ISO 27001. Dữ liệu người dùng được lưu trữ tối đa 2 năm..."},
            {"id": "doc_002", "text": "Quy trình xử lý khiếu nại: Khách hàng có thể gửi khiếu nại qua email hoặc hotline. Thời gian xử lý tối đa 5 ngày làm việc..."},
            {"id": "doc_003", "text": "Hướng dẫn sử dụng hệ thống: Để đăng nhập, người dùng cần nhập email và mật khẩu. Mật khẩu phải có ít nhất 8 ký tự, bao gồm chữ hoa và số..."},
            {"id": "doc_004", "text": "Điều khoản dịch vụ: Người dùng đồng ý không sử dụng dịch vụ cho mục đích bất hợp pháp. Công ty có quyền đình chỉ tài khoản vi phạm..."},
            {"id": "doc_005", "text": "FAQ: Câu hỏi thường gặp về thanh toán, hoàn tiền, và hỗ trợ kỹ thuật..."},
        ]
        
        self.collection.add(
            ids=[doc["id"] for doc in documents],
            documents=[doc["text"] for doc in documents]
        )
        print(f"Indexed {len(documents)} documents into ChromaDB")

    async def _retrieve(self, question: str, top_k: int = 3) -> tuple[List[str], List[str]]:
        """Retrieval: tìm top_k chunks liên quan nhất"""
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k
        )
        retrieved_ids = results["ids"][0]        # List[str] - doc IDs
        contexts = results["documents"][0]       # List[str] - doc texts
        return retrieved_ids, contexts

    async def _generate(self, question: str, contexts: List[str]) -> tuple[str, Dict]:
        """Generation: gọi LLM với retrieved contexts"""
        context_text = "\n\n".join([f"[Tài liệu {i+1}]:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        system_prompt = """Bạn là trợ lý hỗ trợ chuyên nghiệp. 
Chỉ trả lời dựa trên thông tin từ tài liệu được cung cấp. 
Nếu không có thông tin, hãy nói rõ "Tôi không có thông tin về vấn đề này trong tài liệu."
Trả lời ngắn gọn, rõ ràng và chuyên nghiệp."""

        user_prompt = f"""Dựa trên tài liệu sau, hãy trả lời câu hỏi:

{context_text}

Câu hỏi: {question}"""

        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=512
        )
        
        usage = response.usage
        # Tính cost: gpt-4o-mini $0.15/1M input, $0.60/1M output
        cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000
        
        metadata = {
            "model": "gpt-4o-mini",
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "tokens_used": usage.total_tokens,
            "cost_usd": round(cost, 6),
            "sources": []
        }
        return response.choices[0].message.content, metadata

    async def query(self, question: str) -> Dict:
        """
        Main entry point: Retrieve → Generate
        QUAN TRỌNG: trả về 'retrieved_ids' để RetrievalEvaluator tính Hit Rate & MRR
        """
        # Step 1: Retrieve
        retrieved_ids, contexts = await self._retrieve(question, top_k=3)
        
        # Step 2: Generate
        answer, metadata = await self._generate(question, contexts)
        metadata["sources"] = retrieved_ids
        
        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,  # BẮT BUỘC cho Retrieval Eval
            "metadata": metadata
        }
```

**Commit:** `feat(agent): implement RAG agent with ChromaDB retrieval and GPT-4o-mini`

#### Option C — Mock Retrieval với ID Mapping (nếu không có thời gian setup Vector DB):

```python
# Lookup table đơn giản thay thế Vector DB
KNOWLEDGE_BASE = {
    "doc_001": {"text": "Chính sách bảo mật...", "keywords": ["bảo mật", "mã hóa", "GDPR", "dữ liệu"]},
    "doc_002": {"text": "Quy trình khiếu nại...", "keywords": ["khiếu nại", "phản ánh", "5 ngày"]},
    "doc_003": {"text": "Hướng dẫn đăng nhập...", "keywords": ["đăng nhập", "mật khẩu", "tài khoản"]},
    "doc_004": {"text": "Điều khoản...", "keywords": ["điều khoản", "vi phạm", "đình chỉ"]},
    "doc_005": {"text": "FAQ...", "keywords": ["thanh toán", "hoàn tiền", "hỗ trợ"]},
}

def keyword_retrieve(question: str, top_k: int = 3) -> list[str]:
    """Simple keyword matching - thay thế vector search"""
    q_lower = question.lower()
    scores = []
    for doc_id, doc in KNOWLEDGE_BASE.items():
        score = sum(1 for kw in doc["keywords"] if kw in q_lower)
        scores.append((doc_id, score))
    scores.sort(key=lambda x: -x[1])
    return [doc_id for doc_id, _ in scores[:top_k]]
```

---

### [T+1:30 — T+2:30] Phase 3: Wire up Retrieval Evaluator

Implement `evaluate_batch` thực sự trong `engine/retrieval_eval.py`:

```python
from typing import List, Dict

class RetrievalEvaluator:
    def __init__(self, agent=None):
        self.agent = agent  # Inject agent để gọi query

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        top_retrieved = retrieved_ids[:top_k]
        return 1.0 if any(doc_id in top_retrieved for doc_id in expected_ids) else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Tính Hit Rate và MRR thực tế từ agent responses.
        Dataset cần có trường 'expected_retrieval_ids'.
        """
        hit_rates = []
        mrr_scores = []
        
        for case in dataset:
            if "expected_retrieval_ids" not in case:
                continue  # skip cases không có ground truth retrieval
            
            # Gọi agent để lấy retrieved_ids
            if self.agent:
                response = await self.agent.query(case["question"])
                retrieved_ids = response.get("retrieved_ids", [])
            else:
                retrieved_ids = case.get("retrieved_ids", [])  # fallback nếu đã có sẵn
            
            expected_ids = case["expected_retrieval_ids"]
            hit_rates.append(self.calculate_hit_rate(expected_ids, retrieved_ids))
            mrr_scores.append(self.calculate_mrr(expected_ids, retrieved_ids))
        
        n = len(hit_rates) if hit_rates else 1
        return {
            # Tên field phải là "hit_rate" và "mrr" — check_lab.py validate theo tên này
            "hit_rate": round(sum(hit_rates) / n, 4),
            "mrr": round(sum(mrr_scores) / n, 4),
            "total_evaluated": len(hit_rates),
            "hit_rate_per_case": hit_rates,
            "mrr_per_case": mrr_scores
        }
```

**Commit:** `feat(retrieval): wire up evaluate_batch with real agent retrieved_ids`

---

### [T+2:30 — T+3:00] Phase 4: Unit Test & Validate

Test agent trả về đúng format:
```bash
python -c "
import asyncio
from agent.main_agent import MainAgent

async def test():
    agent = MainAgent()
    resp = await agent.query('Chính sách bảo mật dữ liệu là gì?')
    print('Answer:', resp['answer'][:100])
    print('Retrieved IDs:', resp['retrieved_ids'])
    print('Cost:', resp['metadata']['cost_usd'])
    assert 'retrieved_ids' in resp, 'MISSING retrieved_ids!'
    assert len(resp['retrieved_ids']) > 0, 'Empty retrieved_ids!'
    print('✅ Agent format OK')

asyncio.run(test())
"
```

Test retrieval evaluator:
```bash
python -c "
import asyncio, json
from agent.main_agent import MainAgent
from engine.retrieval_eval import RetrievalEvaluator

async def test():
    agent = MainAgent()
    evaluator = RetrievalEvaluator(agent=agent)
    
    # Test với 2 cases mẫu
    test_cases = [
        {'question': 'Chính sách bảo mật?', 'expected_retrieval_ids': ['doc_001']},
        {'question': 'Làm thế nào để khiếu nại?', 'expected_retrieval_ids': ['doc_002']}
    ]
    result = await evaluator.evaluate_batch(test_cases)
    print('Hit Rate:', result['avg_hit_rate'])
    print('MRR:', result['avg_mrr'])

asyncio.run(test())
"
```

**Commit:** `test(agent): validate retrieved_ids format and retrieval eval pipeline`

---

### [T+3:00 — T+3:30] Phase 5: Tích hợp với Pipeline

Phối hợp với Member 1 để kết nối `RetrievalEvaluator` vào `ExpertEvaluator` trong `main.py`:

```python
# Trong ExpertEvaluator.score(), cần tính retrieval từ case + response:
retrieval_eval = RetrievalEvaluator()
hit_rate = retrieval_eval.calculate_hit_rate(
    expected_ids=case.get("expected_retrieval_ids", []),
    retrieved_ids=response.get("retrieved_ids", [])
)
mrr = retrieval_eval.calculate_mrr(
    expected_ids=case.get("expected_retrieval_ids", []),
    retrieved_ids=response.get("retrieved_ids", [])
)
```

- [ ] Xác nhận `retrieved_ids` flow: `agent.query()` → `runner.py` → `evaluator.score()`
- [ ] Không cần gọi agent lại trong evaluator — dùng response đã có từ runner
- [ ] Kiểm tra `reports/summary.json` sau khi chạy có `hit_rate` thực tế (không phải 1.0 hardcode)

**Commit:** `feat(retrieval): integrate hit_rate and mrr into main pipeline`

---

### [T+3:30 — T+4:00] Phase 6: PR & Reflection

- [ ] Tạo PR:
  ```bash
  git push origin feature/member4-agent-retrieval
  gh pr create --title "feat(agent): RAG agent with ChromaDB and retrieval evaluation" \
    --body "## Changes
  - MainAgent with ChromaDB vector store and GPT-4o-mini generation
  - Returns retrieved_ids for Retrieval Evaluation
  - RetrievalEvaluator.evaluate_batch() wired to real agent
  - Cost tracking per query
  - Hit Rate and MRR calculated from real retrieval results"
  ```
- [ ] Viết `analysis/reflections/reflection_Member4.md`

---

## Git Workflow

```bash
git checkout feature/member4-agent-retrieval

# Commit thường xuyên theo milestone
git commit -m "feat(agent): initialize ChromaDB and load document corpus"
git commit -m "feat(agent): implement retrieve() with embedding search"
git commit -m "feat(agent): implement generate() with RAG prompt and cost tracking"
git commit -m "feat(retrieval): wire up evaluate_batch with agent query"
git commit -m "test(agent): validate retrieved_ids format and pipeline integration"

# Sync và push
git fetch origin && git rebase origin/main
git push origin feature/member4-agent-retrieval
```

---

## Kiến thức Cần Nắm (cho phần Điểm Cá nhân)

**Hit Rate @ K:**
- "Có ít nhất 1 document đúng nằm trong top-K kết quả không?"
- Hit Rate @ 3 = 1 nếu bất kỳ expected_id nào nằm trong top-3 retrieved
- Metric đơn giản nhưng không phân biệt được vị trí

**MRR (Mean Reciprocal Rank):**
- Đo vị trí của document đúng đầu tiên: `MRR = 1/rank`
- Rank 1 → MRR = 1.0, Rank 2 → MRR = 0.5, Rank 3 → MRR = 0.33
- Penalize mạnh khi document đúng bị đẩy xuống thấp

**Mối liên hệ Retrieval ↔ Answer Quality:**
- Nếu Hit Rate thấp (< 0.7) → Hallucination cao vì LLM không thấy đúng context
- Nếu MRR thấp nhưng Hit Rate cao → document đúng có trong top-K nhưng bị đẩy xuống dưới → LLM ít chú ý hơn
- Chunking quá lớn → dilute thông tin → MRR thấp

**Embedding Model Choice:**
- `text-embedding-3-small`: $0.02/1M tokens, đủ tốt cho most use cases
- `text-embedding-3-large`: tốt hơn 20%, đắt hơn 5x — không cần cho lab này

---

## Tiêu chí Hoàn thành

- [ ] `agent.query()` trả về `retrieved_ids` là List[str] có ít nhất 1 phần tử
- [ ] `agent.query()` trả về `metadata.cost_usd` (float)
- [ ] `RetrievalEvaluator.evaluate_batch()` tính Hit Rate & MRR từ dữ liệu thực
- [ ] `reports/summary.json` có `hit_rate` khác 0 và không phải hardcode 1.0
- [ ] Corpus có ít nhất 5 documents được index
- [ ] `analysis/reflections/reflection_Member4.md` đã viết
