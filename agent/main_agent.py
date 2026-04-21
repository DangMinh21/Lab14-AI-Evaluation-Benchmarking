import asyncio
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "docs")

CHUNK_SIZE = 250
OVERLAP=25

def load_documents(docs_dir: str) -> List[Dict]:
    """Đọc tất cả file .txt trong data/docs/, chunk theo fixed_size (250 words, overlap 25 words)"""
    # cz = CHUNK_SIZE = 250  # words
    # ol = OVERLAP = 25      # words

    chunks = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(".txt"):
            continue
        doc_id = filename.replace(".txt", "")
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            words = f.read().split()

        # Sliding window chunking
        start = 0
        chunk_idx = 0
        while start < len(words):
            end = start + CHUNK_SIZE
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "id": f"{doc_id}_chunk_{chunk_idx}",
                "doc_id": doc_id,          # giữ lại doc_id gốc để map khi eval
                "text": chunk_text,
            })
            chunk_idx += 1
            start += CHUNK_SIZE - OVERLAP  # slide với overlap

    return chunks


_DEFAULT_SYSTEM_PROMPT = (
    "Bạn là trợ lý hỗ trợ nội bộ. "
    "Chỉ trả lời dựa trên tài liệu được cung cấp. "
    "Nếu không có thông tin, hãy nói rõ: 'Tôi không có thông tin về vấn đề này trong tài liệu.' "
    "Trả lời ngắn gọn, rõ ràng."
)

_V2_SYSTEM_PROMPT = (
    "Bạn là trợ lý hỗ trợ khách hàng chuyên nghiệp và chính xác. "
    "Nhiệm vụ: Trả lời đầy đủ, chi tiết và chính xác dựa HOÀN TOÀN trên tài liệu được cung cấp. "
    "Quy tắc quan trọng: "
    "1. Đọc kỹ TẤT CẢ tài liệu trước khi trả lời. "
    "2. Trích dẫn chính xác: ngày tháng cụ thể, con số chính xác, tên gọi đúng. "
    "3. Nếu câu hỏi về điều kiện hoặc ngoại lệ, liệt kê ĐẦY ĐỦ tất cả trường hợp. "
    "4. Trả lời trực tiếp vào đúng câu hỏi được hỏi, không thêm thông tin lạc đề. "
    "5. Nếu không có thông tin trong tài liệu, trả lời: 'Tôi không có thông tin về vấn đề này trong tài liệu.' "
    "6. Tuyệt đối không bịa thêm thông tin ngoài tài liệu."
)


class MainAgent:
    """RAG Agent sử dụng ChromaDB persistent + OpenAI embeddings"""

    def __init__(self, top_k: int = 3, system_prompt: str = None):
        self.name = "SupportAgent-v1"
        self.top_k = top_k
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
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
                metadatas=[{"doc_id": doc["doc_id"]} for doc in documents],
            )
            print(f"[ChromaDB] Indexed {len(documents)} chunks from {DOCS_DIR}.")
        else:
            print(f"[ChromaDB] Loaded {self.collection.count()} chunks from disk.")

    async def _retrieve(self, question: str, top_k: int = 3) -> tuple[List[str], List[str], List[str]]:
        """Vector search: trả về doc IDs gốc (cho eval), chunk IDs (cho analysis), và chunk texts"""
        results = self.collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents", "metadatas"],
        )
        contexts: List[str] = results["documents"][0]
        chunk_ids: List[str] = results["ids"][0]

        # Map chunk IDs về doc IDs gốc, giữ thứ tự và dedup
        seen = set()
        retrieved_ids: List[str] = []
        for meta in results["metadatas"][0]:
            doc_id = meta["doc_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                retrieved_ids.append(doc_id)

        return retrieved_ids, chunk_ids, contexts

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
                    "content": self.system_prompt,
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
        retrieved_ids, chunk_ids, contexts = await self._retrieve(question, top_k=self.top_k)
        answer, metadata = await self._generate(question, contexts)
        metadata["sources"] = retrieved_ids

        return {
            "answer": answer,
            "contexts": contexts,
            "retrieved_ids": retrieved_ids,        # doc IDs — cho Retrieval Evaluation
            "retrieved_chunk_ids": chunk_ids,      # chunk IDs — cho Failure Analysis
            "metadata": metadata,
        }


if __name__ == "__main__":
    async def test():
        agent = MainAgent()
        question = "Nhân viên mới vào công ty cần làm gì để có thể làm remote?"
        resp = await agent.query(question)
        print("Chunk size: ", CHUNK_SIZE)
        print("Overlap", OVERLAP)
        print("Question", question)
        print("Answer:", resp["answer"])
        print("Retrieved IDs:", resp["retrieved_ids"])
        print("Cost:", resp["metadata"]["cost_usd"])
        assert "retrieved_ids" in resp and len(resp["retrieved_ids"]) > 0
        print("✅ Agent OK")

    asyncio.run(test())
