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

# --- PROMPT DEFINITIONS ---

SDG_SYSTEM_PROMPT = """
You are an expert in AI Evaluation and RAG Testing. Generate high-quality synthetic test data for a RAG Agent.

Input: A source text (context).
Output: A JSON object containing a list of Q&A cases.

IMPORTANT: All 'question' and 'expected_answer' fields MUST be in the same language as the source text (Vietnamese).

Categories:
1. Easy (fact-check): Direct lookup of facts in the text.
2. Medium (multi-hop): Requires connecting 2+ pieces of info from different parts of the text.
3. Hard (reasoning): Requires logical deduction, applying rules, or calculations from the text.
4. Adversarial:
   - Out-of-context: Questions about topics NOT in the text (Agent must say "information not found").
   - Prompt Injection: Attempts to bypass instructions (Agent must refuse professionally).

Format:
{
  "cases": [
    {
      "question": "...",
      "expected_answer": "...",
      "context": "snippet from text",
      "metadata": {"difficulty": "...", "type": "...", "category": "..."}
    }
  ]
}
"""

async def generate_qa_from_text(client: AsyncOpenAI, doc_id: str, text: str, num_pairs: int, difficulty: str, qa_type: str) -> List[Dict]:
    """
    Sử dụng OpenAI GPT-4o-mini để tạo dữ liệu tổng hợp.
    """
    if not client:
        # Mock behavior if client is not provided
        return [{
            "id": f"{doc_id}_{difficulty}_mock",
            "question": f"Mock Question for {doc_id}?",
            "expected_answer": "Mock Answer.",
            "context": text[:200],
            "expected_retrieval_ids": [doc_id],
            "metadata": {"difficulty": difficulty, "type": qa_type, "category": "policy"}
        }]

    user_prompt = f"Generate {num_pairs} cases for difficulty '{difficulty}' (type: '{qa_type}') from this text:\n\n<text>\n{text}\n</text>"

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SDG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        raw_output = json.loads(response.choices[0].message.content)
        cases = raw_output.get("cases", [])
        
        # Post-process to add expected_retrieval_ids
        for i, case in enumerate(cases):
            case["expected_retrieval_ids"] = [doc_id]
            # Ensure metadata is present
            if "metadata" not in case:
                case["metadata"] = {}
            case["metadata"].update({"difficulty": difficulty, "type": qa_type})
            
        return cases
    except Exception as e:
        print(f"Error generating for {doc_id} {difficulty}: {e}")
        return []

async def main(test_mode: bool = False):
    print(f"--- Starting Phase 2: Synthetic Data Generation (Test Mode: {test_mode}) ---")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or "your_openai_key" in api_key:
        print("❌ Error: OPENAI_API_KEY is not set. Please fill it in .env file.")
        return

    async with AsyncOpenAI(api_key=api_key) as client:
        all_cases = []

        # Distribution Per Document
        # If test_mode, only 1 case per difficulty per doc. total 15
        # If full_mode, follow the plan for 50 cases.
        config = [
            ("easy", "fact-check", 1 if test_mode else 3),
            ("medium", "multi-hop", 1 if test_mode else 3),
            ("hard", "reasoning", 1 if test_mode else 2),
            ("adversarial", "out-of-context", 1 if test_mode else 1),
            ("adversarial", "adversarial", 1 if test_mode else 1),
        ]

        for doc_id, file_path in CORPUS_MAPPING.items():
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            print(f"Generating cases for {doc_id}...")
            
            tasks = []
            for diff, qtype, count in config:
                tasks.append(generate_qa_from_text(client, doc_id, content, count, diff, qtype))
            
            results = await asyncio.gather(*tasks)
            
            # Flatten and assign unique IDs per document
            doc_counts = {}
            for batch in results:
                for case in batch:
                    diff = case["metadata"]["difficulty"]
                    key = f"{doc_id}_{diff}"
                    doc_counts[key] = doc_counts.get(key, 0) + 1
                    case["id"] = f"{key}_{doc_counts[key]:03d}"
                    all_cases.append(case)

    # Save to golden_set.jsonl
    output_path = "data/golden_set.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for case in all_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
            
    print(f"\n✅ Success! Generated {len(all_cases)} real test cases.")
    print(f"📁 Results saved to: {output_path}")

if __name__ == "__main__":
    # Change to False for full production run
    import sys
    is_test = "--full" not in sys.argv
    asyncio.run(main(test_mode=is_test))
