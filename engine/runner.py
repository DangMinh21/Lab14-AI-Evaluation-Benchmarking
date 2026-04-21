import asyncio
import time
import numpy as np
from typing import List, Dict
from tqdm.asyncio import tqdm


class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()

        response = await self.agent.query(test_case["question"])
        latency = round(time.perf_counter() - start_time, 4)

        # Inject retrieved_ids vào case để evaluator dùng — không gọi agent lần 2
        case_with_retrieval = {**test_case, "retrieved_ids": response.get("retrieved_ids", [])}

        ragas_scores = await self.evaluator.score(case_with_retrieval, response)

        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case["expected_answer"]
        )

        cost_usd = response.get("metadata", {}).get("cost_usd", 0.0)

        return {
            "test_case": test_case["question"],
            "case_id": test_case.get("id", ""),
            "difficulty": test_case.get("metadata", {}).get("difficulty", "unknown"),
            "agent_response": response["answer"],
            "retrieved_ids": response.get("retrieved_ids", []),
            "latency": latency,
            "cost_usd": cost_usd,
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3 else "pass"
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        """Chạy song song theo batch, giới hạn concurrency để tránh rate limit."""
        results = []
        total_batches = (len(dataset) + batch_size - 1) // batch_size

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} cases)...")
            tasks = [self.run_single_test(case) for case in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results

    @staticmethod
    def compute_stats(results: List[Dict]) -> Dict:
        """Tính aggregate stats từ kết quả: latency percentiles, cost, pass rate."""
        latencies = [r["latency"] for r in results]
        costs = [r["cost_usd"] for r in results]
        scores = [r["judge"]["final_score"] for r in results]

        total = len(results)
        passed = sum(1 for r in results if r["status"] == "pass")

        return {
            "total": total,
            "pass_count": passed,
            "fail_count": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0,
            "avg_score": round(sum(scores) / total, 4) if total else 0,
            "p50_latency": round(float(np.percentile(latencies, 50)), 3),
            "p95_latency": round(float(np.percentile(latencies, 95)), 3),
            "total_cost_usd": round(sum(costs), 6),
            "avg_cost_per_case": round(sum(costs) / total, 6) if total else 0,
        }
