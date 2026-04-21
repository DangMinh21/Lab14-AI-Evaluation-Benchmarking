import asyncio
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "docs")


def load_documents(docs_dir: str) -> List[Dict]:
    """Đọc tất cả file .txt trong data/docs/, dùng tên file (không có .txt) làm doc ID"""
    docs = []
    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith(".txt"):
            doc_id = filename.replace(".txt", "")
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                docs.append({"id": doc_id, "text": f.read()})
    return docs


class MainAgent:
    """RAG Agent sử dụng ChromaDB in-memory + OpenAI embeddings"""

    def __init__(self):
        self.name = "SupportAgent-v1"
        self._init_vector_store()

    def _init_vector_store(self):
        """Khởi tạo ChromaDB persistent và index toàn bộ corpus từ data/docs/"""
        import chromadb
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

        self.chroma_client = chromadb.PersistentClient(path="./data/chromadb")

        embedding_fn = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=embedding_fn,
        )

        # Chỉ index nếu collection còn trống
        if self.collection.count() == 0:
            documents = load_documents(DOCS_DIR)
            self.collection.add(
                ids=[doc["id"] for doc in documents],
                documents=[doc["text"] for doc in documents],
            )
            print(f"[ChromaDB] Indexed {len(documents)} documents from {DOCS_DIR}.")
        else:
            print(f"[ChromaDB] Loaded {self.collection.count()} documents from disk.")

    async def _retrieve(self, question: str, top_k: int = 3) -> tuple[List[str], List[str]]:
        """Vector search: trả về top_k doc IDs và texts liên quan nhất"""
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
        )
        retrieved_ids: List[str] = results["ids"][0]
        contexts: List[str] = results["documents"][0]
        return retrieved_ids, contexts

    async def _generate(self, question: str, contexts: List[str]) -> tuple[str, Dict]:
        """Gọi OpenAI với retrieved contexts để sinh câu trả lời"""
        from openai import AsyncOpenAI

        llm = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        context_text = "\n\n".join(
            f"[Tài liệu {i + 1}]:\n{ctx}" for i, ctx in enumerate(contexts)
        )

        response = await llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý hỗ trợ nội bộ. "
                        "Chỉ trả lời dựa trên tài liệu được cung cấp. "
                        "Nếu không có thông tin, hãy nói rõ: 'Tôi không có thông tin về vấn đề này trong tài liệu.' "
                        "Trả lời ngắn gọn, rõ ràng."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Dựa trên tài liệu sau:\n\n{context_text}\n\nCâu hỏi: {question}",
                },
            ],
            temperature=0.1,
            max_tokens=512,
        )

        usage = response.usage
        # gpt-4o-mini: $0.15/1M input tokens, $0.60/1M output tokens
        cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000

        metadata = {
            "model": "gpt-4o-mini",
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "tokens_used": usage.total_tokens,
            "cost_usd": round(cost, 6),
        }
        return response.choices[0].message.content, metadata

    async def query(self, question: str) -> Dict:
        """
        RAG pipeline: Retrieve → Generate
        Trả về retrieved_ids để RetrievalEvaluator tính Hit Rate & MRR.
        """
        retrieved_ids, contexts = await self._retrieve(question, top_k=3)
        answer, metadata = await self._generate(question, contexts)
        metadata["sources"] = retrieved_ids

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,   # BẮT BUỘC cho Retrieval Evaluation
            "metadata": metadata,
        }


if __name__ == "__main__":
    async def test():
        agent = MainAgent()
        resp = await agent.query("Chính sách nghỉ phép năm là gì?")
        print("Answer:", resp["answer"][:150])
        print("Retrieved IDs:", resp["retrieved_ids"])
        print("Cost:", resp["metadata"]["cost_usd"])
        assert "retrieved_ids" in resp and len(resp["retrieved_ids"]) > 0
        print("✅ Agent OK")

    asyncio.run(test())
