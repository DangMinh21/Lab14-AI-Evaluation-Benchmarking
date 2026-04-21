import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge
from agent.main_agent import MainAgent

_retrieval_eval = RetrievalEvaluator()


class ExpertEvaluator:
    async def score(self, case, resp):
        faithfulness_score = 0.0
        relevancy_score = 0.0
        try:
            from ragas import evaluate, EvaluationDataset, SingleTurnSample
            from ragas.metrics import faithfulness, answer_relevancy

            sample = SingleTurnSample(
                user_input=case["question"],
                response=resp["answer"],
                retrieved_contexts=resp.get("contexts", []),
                reference=case["expected_answer"],
            )
            result = evaluate(EvaluationDataset(samples=[sample]), metrics=[faithfulness, answer_relevancy])
            row = result.to_pandas().iloc[0]
            faithfulness_score = float(row.get("faithfulness", 0) or 0)
            relevancy_score = float(row.get("answer_relevancy", 0) or 0)
        except Exception as e:
            print(f"    [RAGAS] Warning: {e}")

        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = case.get("retrieved_ids", [])
        return {
            "faithfulness": round(faithfulness_score, 4),
            "relevancy": round(relevancy_score, 4),
            "retrieval": {
                "hit_rate": _retrieval_eval.calculate_hit_rate(expected_ids, retrieved_ids),
                "mrr": _retrieval_eval.calculate_mrr(expected_ids, retrieved_ids),
            },
        }


class MultiModelJudge:
    def __init__(self):
        self._judge = LLMJudge()

    async def evaluate_multi_judge(self, q, a, gt):
        return await self._judge.evaluate_multi_judge(q, a, gt)

async def run_benchmark_with_results(agent_version: str):
    print(f"🚀 Khởi động Benchmark cho {agent_version}...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng. Hãy tạo ít nhất 1 test case.")
        return None, None

    runner = BenchmarkRunner(MainAgent(), ExpertEvaluator(), MultiModelJudge())
    results = await runner.run_all(dataset)

    total = len(results)
    summary = {
        "metadata": {"version": agent_version, "total": total, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total
        }
    }
    return results, summary

async def run_benchmark(version):
    _, summary = await run_benchmark_with_results(version)
    return summary

async def main():
    v1_summary = await run_benchmark("Agent_V1_Base")
    
    # Giả lập V2 có cải tiến (để test logic)
    v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    
    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_summary["metrics"]["avg_score"] - v1_summary["metrics"]["avg_score"]
    print(f"V1 Score: {v1_summary['metrics']['avg_score']}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']}")
    print(f"Delta: {'+' if delta >= 0 else ''}{delta:.2f}")

    # Multi-threshold Release Gate
    THRESHOLDS = {
        "min_avg_score":      3.5,
        "min_hit_rate":       0.70,
        "min_agreement_rate": 0.60,
        "max_latency_p95":    5.0,
        "max_cost_per_case":  0.01,
    }
    m = v2_summary["metrics"]
    gate_checks = {
        "score_improved":  delta >= 0,
        "score_threshold": m["avg_score"] >= THRESHOLDS["min_avg_score"],
        "retrieval_ok":    m["hit_rate"] >= THRESHOLDS["min_hit_rate"],
        "judge_consensus": m["agreement_rate"] >= THRESHOLDS["min_agreement_rate"],
        "latency_ok":      m.get("p95_latency", 0) <= THRESHOLDS["max_latency_p95"],
        "cost_ok":         m.get("avg_cost_per_case", 0) <= THRESHOLDS["max_cost_per_case"],
    }
    failed_checks = [k for k, v in gate_checks.items() if not v]
    v2_summary["regression"] = {
        "delta_score": round(delta, 4),
        "checks": gate_checks,
        "failed_checks": failed_checks,
    }
    if failed_checks:
        print(f"⚠️  Failed checks: {failed_checks}")

    os.makedirs("reports", exist_ok=True)
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, ensure_ascii=False, indent=2)
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(v2_results, f, ensure_ascii=False, indent=2)

    if delta > 0:
        print("✅ QUYẾT ĐỊNH: CHẤP NHẬN BẢN CẬP NHẬT (APPROVE)")
    else:
        print("❌ QUYẾT ĐỊNH: TỪ CHỐI (BLOCK RELEASE)")

if __name__ == "__main__":
    asyncio.run(main())
