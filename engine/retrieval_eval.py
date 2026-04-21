from typing import List, Dict


class RetrievalEvaluator:
    def __init__(self):
        pass

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
        Tính Hit Rate & MRR từ retrieved_ids đã có sẵn trong mỗi case
        (được inject bởi runner sau khi agent.query() trả về).
        """
        hit_rates = []
        mrr_scores = []

        for case in dataset:
            expected_ids = case.get("expected_retrieval_ids", [])
            retrieved_ids = case.get("retrieved_ids", [])
            if not expected_ids:
                continue
            hit_rates.append(self.calculate_hit_rate(expected_ids, retrieved_ids))
            mrr_scores.append(self.calculate_mrr(expected_ids, retrieved_ids))

        n = len(hit_rates) if hit_rates else 1
        return {
            "hit_rate": round(sum(hit_rates) / n, 4),
            "mrr": round(sum(mrr_scores) / n, 4),
            "total_evaluated": len(hit_rates),
        }
