import json
with open('data/golden_set.jsonl', 'r', encoding='utf-8') as f:
    cases = [json.loads(l) for l in f if l.strip()]
adv = [c for c in cases if c.get('metadata', {}).get('difficulty') == 'adversarial']
print(f"Total adversarial cases: {len(adv)}")
print(f"All have empty retrieval: {all(c.get('expected_retrieval_ids') == [] for c in adv)}")
