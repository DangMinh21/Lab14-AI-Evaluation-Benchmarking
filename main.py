import asyncio
import json
import os
import time
from engine.runner import BenchmarkRunner
from engine.retrieval_eval import RetrievalEvaluator
from engine.llm_judge import LLMJudge
from agent.main_agent import MainAgent, _DEFAULT_SYSTEM_PROMPT, _V2_SYSTEM_PROMPT

_retrieval_eval = RetrievalEvaluator()


class ExpertEvaluator:
    async def score(self, case, resp):
        faithfulness_score = 0.0
        relevancy_score = 0.0
        try:
            from ragas import evaluate, EvaluationDataset, SingleTurnSample
            from ragas.metrics.collections import faithfulness

            sample = SingleTurnSample(
                user_input=case["question"],
                response=resp["answer"],
                retrieved_contexts=resp.get("contexts", []),
                reference=case["expected_answer"],
            )
            result = evaluate(EvaluationDataset(samples=[sample]), metrics=[faithfulness])
            row = result.to_pandas().iloc[0]
            faithfulness_score = float(row.get("faithfulness", 0) or 0)
            relevancy_score = 0.0  # answer_relevancy bị lỗi embed_query trong ragas 0.4.x
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


def to_template_format(result: dict) -> dict:
    """Transform runner output to match the expected template format for auto-grading."""
    ragas = result["ragas"]
    retrieval = ragas.get("retrieval", {})
    judge = result["judge"]

    individual_results = {
        model_name: {
            "score": judgment.get("overall_score", 0),
            "reasoning": judgment.get("reasoning", ""),
        }
        for model_name, judgment in judge.get("individual_judgments", {}).items()
    }

    score_gap = judge.get("score_gap", 0)

    return {
        "test_case": result["test_case"],
        "agent_response": result["agent_response"],
        "latency": result["latency"],
        "ragas": {
            "hit_rate": retrieval.get("hit_rate", 0.0),
            "mrr": retrieval.get("mrr", 0.0),
            "faithfulness": ragas.get("faithfulness", 0.0),
            "relevancy": ragas.get("relevancy", 0.0),
        },
        "judge": {
            "final_score": judge["final_score"],
            "agreement_rate": judge["agreement_rate"],
            "individual_results": individual_results,
            "status": "conflict" if score_gap > 1 else "consensus",
        },
        "status": result["status"],
    }


async def run_benchmark_with_results(agent_version: str, top_k: int = 3, system_prompt: str = None):
    print(f"\n🚀 Khởi động Benchmark cho {agent_version} (top_k={top_k})...")

    if not os.path.exists("data/golden_set.jsonl"):
        print("❌ Thiếu data/golden_set.jsonl. Hãy chạy 'python data/synthetic_gen.py' trước.")
        return None, None

    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    if not dataset:
        print("❌ File data/golden_set.jsonl rỗng.")
        return None, None

    print(f"📂 Dataset: {len(dataset)} test cases")

    agent = MainAgent(top_k=top_k, system_prompt=system_prompt)
    runner = BenchmarkRunner(agent, ExpertEvaluator(), MultiModelJudge())
    results = await runner.run_all(dataset)

    stats = BenchmarkRunner.compute_stats(results)

    retrieval_inputs = [
        {**case, "retrieved_ids": r["retrieved_ids"]}
        for case, r in zip(dataset, results)
    ]
    retrieval_metrics = await _retrieval_eval.evaluate_batch(retrieval_inputs)

    total = len(results)
    agreement_rate = round(
        sum(r["judge"]["agreement_rate"] for r in results) / total, 4
    )

    summary = {
        "version": agent_version,
        "metrics": {
            **stats,
            "hit_rate": retrieval_metrics["hit_rate"],
            "mrr": retrieval_metrics["mrr"],
            "retrieval_evaluated": retrieval_metrics["total_evaluated"],
            "agreement_rate": agreement_rate,
        },
    }
    return results, summary


async def main():
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # V1: baseline — top_k=3, terse prompt (1-2 câu, bỏ qua chi tiết)
    v1_results, v1_summary = await run_benchmark_with_results(
        "Agent_V1_Base", top_k=3, system_prompt=_DEFAULT_SYSTEM_PROMPT
    )

    # V2: improved — top_k=5, detailed prompt (liệt kê đầy đủ, trích dẫn chính xác)
    v2_results, v2_summary = await run_benchmark_with_results(
        "Agent_V2_Optimized", top_k=5, system_prompt=_V2_SYSTEM_PROMPT
    )

    if not v1_summary or not v2_summary:
        print("❌ Không thể chạy Benchmark. Kiểm tra lại data/golden_set.jsonl.")
        return

    v1_m = v1_summary["metrics"]
    v2_m = v2_summary["metrics"]

    print("\n📊 --- KẾT QUẢ SO SÁNH (REGRESSION) ---")
    delta = v2_m["avg_score"] - v1_m["avg_score"]
    print(f"V1 Score: {v1_m['avg_score']:.4f}  |  V2 Score: {v2_m['avg_score']:.4f}  |  Delta: {'+' if delta >= 0 else ''}{delta:.4f}")
    print(f"Hit Rate — V1: {v1_m['hit_rate']*100:.1f}%  V2: {v2_m['hit_rate']*100:.1f}%")
    print(f"P95 Latency — V1: {v1_m['p95_latency']:.2f}s  V2: {v2_m['p95_latency']:.2f}s")
    print(f"Avg Cost/case — V1: ${v1_m['avg_cost_per_case']:.6f}  V2: ${v2_m['avg_cost_per_case']:.6f}")
    print(f"Pass Rate — V1: {v1_m['pass_rate']*100:.1f}%  V2: {v2_m['pass_rate']*100:.1f}%")

    THRESHOLDS = {
        "min_avg_score":      3.5,
        "min_hit_rate":       0.70,
        "min_agreement_rate": 0.60,
        "max_latency_p95":    10.0,   # realistic for async RAG + judge pipeline
        "max_cost_per_case":  0.01,
    }
    gate_checks = {
        "score_improved":  delta >= 0,
        "score_threshold": v2_m["avg_score"] >= THRESHOLDS["min_avg_score"],
        "retrieval_ok":    v2_m["hit_rate"] >= THRESHOLDS["min_hit_rate"],
        "judge_consensus": v2_m["agreement_rate"] >= THRESHOLDS["min_agreement_rate"],
        "latency_ok":      v2_m["p95_latency"] <= THRESHOLDS["max_latency_p95"],
        "cost_ok":         v2_m["avg_cost_per_case"] <= THRESHOLDS["max_cost_per_case"],
    }
    failed_checks = [k for k, v in gate_checks.items() if not v]
    decision = "APPROVE" if not failed_checks else "BLOCK"

    print("\n🔒 --- RELEASE GATE ---")
    for check, passed in gate_checks.items():
        print(f"  {'✅' if passed else '❌'} {check}")
    print(f"\n{'✅' if decision == 'APPROVE' else '❌'} QUYẾT ĐỊNH: {decision}")

    # ── Save outputs in template-matching format ──────────────────────────────
    os.makedirs("reports", exist_ok=True)

    benchmark_results = {
        "v1": [to_template_format(r) for r in v1_results],
        "v2": [to_template_format(r) for r in v2_results],
    }
    with open("reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    summary_out = {
        "metadata": {
            "total": len(v1_results),
            "version": "BASELINE (V1)",
            "timestamp": timestamp,
            "versions_compared": ["V1", "V2"],
        },
        "metrics": {
            "avg_score": round(v1_m["avg_score"], 4),
            "hit_rate": round(v1_m["hit_rate"], 4),
            "agreement_rate": round(v1_m["agreement_rate"], 4),
        },
        "regression": {
            "v1": {
                "score": round(v1_m["avg_score"], 4),
                "hit_rate": round(v1_m["hit_rate"], 4),
                "judge_agreement": round(v1_m["agreement_rate"], 4),
            },
            "v2": {
                "score": round(v2_m["avg_score"], 4),
                "hit_rate": round(v2_m["hit_rate"], 4),
                "judge_agreement": round(v2_m["agreement_rate"], 4),
            },
            "decision": decision,
        },
    }
    with open("reports/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_out, f, ensure_ascii=False, indent=2)

    print(f"\n📁 Đã lưu: reports/summary.json & reports/benchmark_results.json")


if __name__ == "__main__":
    asyncio.run(main())
