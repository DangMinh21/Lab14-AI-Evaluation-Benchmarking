import json
try:
    with open('data/golden_set.jsonl', 'r', encoding='utf-8') as f:
        cases = [json.loads(l) for l in f if l.strip()]
    print(f'Total cases: {len(cases)}')
    print(f'All have retrieval_ids: {all("expected_retrieval_ids" in c for c in cases)}')
    print(f'All have expected_answer: {all("expected_answer" in c for c in cases)}')
    print(f'All have IDs: {all("id" in c for c in cases)}')
    
    # Check for uniqueness of IDs
    ids = [c['id'] for c in cases]
    print(f'All IDs are unique: {len(ids) == len(set(ids))}')
    
    # Print the last few IDs to verify
    for c in cases[:3]:
        print(f"Sample ID: {c['id']}")
    for c in cases[-3:]:
        print(f"Sample ID: {c['id']}")
        
except Exception as e:
    print(f"Validation failed: {e}")
