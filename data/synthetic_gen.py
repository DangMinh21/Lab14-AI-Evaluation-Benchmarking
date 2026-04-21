import json
import asyncio
import os
from typing import List, Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# Load environment variables
load_dotenv()

# Dictionary mapping Document IDs to their file paths
CORPUS_MAPPING = {
    "doc_001": "data/docs/doc_001.txt",
    "doc_002": "data/docs/doc_002.txt",
    "doc_003": "data/docs/doc_003.txt",
    "doc_004": "data/docs/doc_004.txt",
    "doc_005": "data/docs/doc_005.txt",
}

# --- SCHEMA DEFINITION ---
# Each test case must follow this structure:
# {
#   "id": "doc_001_easy_001",
#   "question": "...",
#   "expected_answer": "...",
#   "context": "...",
#   "expected_retrieval_ids": ["doc_001"],
#   "metadata": {
#     "difficulty": "easy|medium|hard|adversarial",
#     "type": "fact-check|multi-hop|out-of-context|adversarial|ambiguous|conflicting",
#     "category": "policy|procedure|technical|safety"
#   }
# }

async def generate_qa_from_text(client: AsyncOpenAI, doc_id: str, text: str, num_pairs: int, difficulty: str, qa_type: str) -> List[Dict]:
    """
    Sử dụng LLM để tạo dữ liệu tổng hợp dựa trên văn bản nguồn.
    Phase 2: Member 2 sẽ hoàn thiện Prompt Engineering tại đây.
    """
    # TODO: Implement complex prompt for different difficulties/types
    
    # Mock behavior for Phase 1 validation
    mock_cases = []
    for i in range(num_pairs):
        mock_cases.append({
            "id": f"{doc_id}_{difficulty}_{i+1:03d}",
            "question": f"Câu hỏi mẫu cho {doc_id} - {difficulty}?",
            "expected_answer": "Câu trả lời mẫu.",
            "context": text[:300],
            "expected_retrieval_ids": [doc_id],
            "metadata": {
                "difficulty": difficulty,
                "type": qa_type,
                "category": "technical" # placeholder
            }
        })
    return mock_cases

async def main():
    print("--- Starting Phase 1: SDG Setup & Validation ---")
    
    # Initialize OpenAI Client (Requires OPENAI_API_KEY in .env)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or "your_openai_key" in api_key:
        print("⚠️ Warning: OPENAI_API_KEY is not set or using placeholder. Running in MOCK mode.")
        client = None
    else:
        client = AsyncOpenAI(api_key=api_key)

    all_cases = []
    
    # Iterate through our defined corpus
    for doc_id, file_path in CORPUS_MAPPING.items():
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        print(f"Processing {doc_id} ({os.path.basename(file_path)})...")
        
        # In a real run, we would call LLM for different categories.
        # For Phase 1 validation, we generate 1 mock case per difficulty.
        for diff, qtype in [("easy", "fact-check"), ("medium", "multi-hop"), ("hard", "reasoning")]:
            cases = await generate_qa_from_text(client, doc_id, content, 1, diff, qtype)
            all_cases.extend(cases)

    # Save to golden_set.jsonl
    os.makedirs("data", exist_ok=True)
    output_path = "data/golden_set.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for case in all_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
            
    print(f"\nPhase 1 complete! Generated {len(all_cases)} mock cases.")
    print(f"Results saved to: {output_path}")
    print("\n--- Next Steps for Member 2 ---")
    print("1. Fill real API keys in .env")
    print("2. Update 'generate_qa_from_text' with real LLM prompts (Phase 2)")
    print("3. Inform Member 4 about the doc_id mapping in CORPUS_MAPPING.")

if __name__ == "__main__":
    asyncio.run(main())
