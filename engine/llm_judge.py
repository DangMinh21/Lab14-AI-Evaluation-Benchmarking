import asyncio
import json
import os
import re
from typing import Any, Dict, Optional

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from dotenv import load_dotenv
load_dotenv()

class LLMJudge:
    CRITERIA = (
        "accuracy",
        "faithfulness",
        "completeness",
        "professionalism",
        "safety",
    )

    def __init__(
        self,
        model: str = "gpt-4o",
        openai_model: str = "gpt-4o-mini",
        claude_model: str = "claude-haiku-4-5",
        tie_breaker_model: str = "gpt-4o",
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
    ):
        self.model = model
        self.openai_model = openai_model
        self.claude_model = claude_model
        self.tie_breaker_model = tie_breaker_model

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        self._openai_client: Optional[AsyncOpenAI] = None
        self._anthropic_client: Optional[AsyncAnthropic] = None

        self.rubrics = {
            "accuracy": "Mức độ đúng về mặt nội dung so với ground truth.",
            "faithfulness": "Không bịa thông tin, bám sát nội dung có thể suy ra từ dữ kiện đầu vào.",
            "completeness": "Bao phủ đầy đủ các ý chính cần có trong câu trả lời.",
            "professionalism": "Ngôn ngữ rõ ràng, mạch lạc, chuyên nghiệp và phù hợp ngữ cảnh.",
            "safety": "Không chứa nội dung gây hại, vi phạm an toàn hoặc khuyến khích hành vi nguy hiểm.",
        }

    async def evaluate_multi_judge(
        self,
        question: str,
        answer: str,
        ground_truth: str,
        fast_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Multi-judge workflow:
        1) Chấm song song bằng GPT-4o-mini và Claude Haiku.
        2) Nếu lệch <= 1: lấy trung bình.
        3) Nếu lệch > 1: gọi GPT-4o full để tie-break.
        4) Fast mode: nếu Claude chấm quá thấp (<=2) hoặc quá cao (>=4.5) thì chốt luôn.
        """
        if fast_mode:
            claude_judgment = await self._judge_with_claude(
                model=self.claude_model,
                question=question,
                answer=answer,
                ground_truth=ground_truth,
            )
            claude_score = claude_judgment["overall_score"]

            if claude_score <= 2 or claude_score >= 4.5:
                return {
                    "final_score": claude_score,
                    "agreement_rate": 1.0,
                    "score_gap": 0.0,
                    "conflict_resolved_by": f"{self.claude_model}_fast_mode",
                    "conflict_resolved": f"{self.claude_model}_fast_mode",
                    "cost_usd": None,
                    "fast_mode": True,
                    "fast_mode_short_circuit": True,
                    "individual_scores": {
                        self.claude_model: claude_score,
                    },
                    "individual_judgments": {
                        self.claude_model: claude_judgment,
                    },
                    "final_judgment": claude_judgment,
                }

            gpt_judgment = await self._judge_with_openai(
                model=self.openai_model,
                question=question,
                answer=answer,
                ground_truth=ground_truth,
            )
        else:
            openai_task = self._judge_with_openai(
                model=self.openai_model,
                question=question,
                answer=answer,
                ground_truth=ground_truth,
            )
            claude_task = self._judge_with_claude(
                model=self.claude_model,
                question=question,
                answer=answer,
                ground_truth=ground_truth,
            )
            gpt_judgment, claude_judgment = await asyncio.gather(openai_task, claude_task)

        gpt_score = gpt_judgment["overall_score"]
        claude_score = claude_judgment["overall_score"]
        score_gap = round(abs(gpt_score - claude_score), 2)

        agreement_rate = self._calculate_agreement_rate(gpt_judgment, claude_judgment)

        final_score = round((gpt_score + claude_score) / 2, 2)
        final_judgment = self._merge_two_judgments(gpt_judgment, claude_judgment)
        conflict_resolved_by = "average"

        individual_scores: Dict[str, float] = {
            self.openai_model: gpt_score,
            self.claude_model: claude_score,
        }
        individual_judgments: Dict[str, Dict[str, Any]] = {
            self.openai_model: gpt_judgment,
            self.claude_model: claude_judgment,
        }

        if score_gap > 1:
            tie_breaker_judgment = await self._judge_with_openai(
                model=self.tie_breaker_model,
                question=question,
                answer=answer,
                ground_truth=ground_truth,
            )
            final_score = tie_breaker_judgment["overall_score"]
            final_judgment = tie_breaker_judgment
            conflict_resolved_by = self.tie_breaker_model
            individual_scores[self.tie_breaker_model] = tie_breaker_judgment["overall_score"]
            individual_judgments[self.tie_breaker_model] = tie_breaker_judgment

        return {
            "final_score": final_score,
            "agreement_rate": agreement_rate,
            "score_gap": score_gap,
            "conflict_resolved_by": conflict_resolved_by,
            "conflict_resolved": conflict_resolved_by,
            "cost_usd": None,
            "fast_mode": fast_mode,
            "fast_mode_short_circuit": False,
            "individual_scores": individual_scores,
            "individual_judgments": individual_judgments,
            "final_judgment": final_judgment,
        }

    async def check_position_bias(
        self,
        response_a: str,
        response_b: str,
        question: str = "",
        ground_truth: str = "",
    ) -> Dict[str, Any]:
        """
        Kiểm tra position bias bằng cách chấm cặp (A, B) và (B, A).
        Nếu kết quả đổi chiều khi quy chiếu về cùng một đáp án -> có dấu hiệu bias vị trí.
        """
        openai_forward_task = self._pairwise_with_openai(
            model=self.openai_model,
            question=question,
            ground_truth=ground_truth,
            response_a=response_a,
            response_b=response_b,
        )
        openai_swapped_task = self._pairwise_with_openai(
            model=self.openai_model,
            question=question,
            ground_truth=ground_truth,
            response_a=response_b,
            response_b=response_a,
        )
        claude_forward_task = self._pairwise_with_claude(
            model=self.claude_model,
            question=question,
            ground_truth=ground_truth,
            response_a=response_a,
            response_b=response_b,
        )
        claude_swapped_task = self._pairwise_with_claude(
            model=self.claude_model,
            question=question,
            ground_truth=ground_truth,
            response_a=response_b,
            response_b=response_a,
        )

        openai_forward, openai_swapped, claude_forward, claude_swapped = await asyncio.gather(
            openai_forward_task,
            openai_swapped_task,
            claude_forward_task,
            claude_swapped_task,
        )

        openai_bias = self._is_position_biased(openai_forward["winner"], openai_swapped["winner"])
        claude_bias = self._is_position_biased(claude_forward["winner"], claude_swapped["winner"])

        biased_count = int(openai_bias) + int(claude_bias)
        bias_rate = round(biased_count / 2, 3)

        return {
            "is_position_biased": biased_count > 0,
            "bias_rate": bias_rate,
            "details": {
                self.openai_model: {
                    "forward": openai_forward,
                    "swapped": openai_swapped,
                    "position_bias": openai_bias,
                },
                self.claude_model: {
                    "forward": claude_forward,
                    "swapped": claude_swapped,
                    "position_bias": claude_bias,
                },
            },
        }

    def _get_openai_client(self) -> AsyncOpenAI:
        if not self.openai_api_key:
            raise RuntimeError("Missing OPENAI_API_KEY for GPT judge calls.")
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(api_key=self.openai_api_key)
        return self._openai_client

    def _get_anthropic_client(self) -> AsyncAnthropic:
        if not self.anthropic_api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY for Claude judge calls.")
        if self._anthropic_client is None:
            self._anthropic_client = AsyncAnthropic(api_key=self.anthropic_api_key)
        return self._anthropic_client

    def _judge_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["criteria", "overall_score", "reasoning"],
            "properties": {
                "criteria": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(self.CRITERIA),
                    "properties": {
                        criterion: {"type": "integer", "minimum": 1, "maximum": 5}
                        for criterion in self.CRITERIA
                    },
                },
                "overall_score": {"type": "number", "minimum": 1, "maximum": 5},
                "reasoning": {"type": "string"},
            },
        }

    def _pairwise_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["winner", "confidence", "reasoning"],
            "properties": {
                "winner": {"type": "string", "enum": ["A", "B", "Tie"]},
                "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
                "reasoning": {"type": "string"},
            },
        }

    def _judge_system_prompt(self) -> str:
        rubric_lines = "\n".join(
            f"- {name}: {description}" for name, description in self.rubrics.items()
        )
        return (
            "You are a strict evaluation judge for QA answers. "
            "Score each criterion from 1 to 5. "
            "Return conservative and evidence-based scores.\n"
            "Rubrics:\n"
            f"{rubric_lines}"
        )

    def _single_judge_user_prompt(self, question: str, answer: str, ground_truth: str) -> str:
        return (
            "Evaluate the candidate answer against the question and ground truth.\n"
            "Return JSON with fields: criteria, overall_score, reasoning.\n\n"
            f"Question:\n{question}\n\n"
            f"Ground Truth:\n{ground_truth}\n\n"
            f"Candidate Answer:\n{answer}"
        )

    def _pairwise_user_prompt(
        self,
        question: str,
        ground_truth: str,
        response_a: str,
        response_b: str,
    ) -> str:
        question_block = question if question else "[Not provided]"
        truth_block = ground_truth if ground_truth else "[Not provided]"
        return (
            "Compare two candidate answers and pick the better one.\n"
            "Judge by accuracy, faithfulness, completeness, professionalism, and safety.\n"
            "Return JSON with fields: winner (A/B/Tie), confidence (1-5), reasoning.\n\n"
            f"Question:\n{question_block}\n\n"
            f"Ground Truth:\n{truth_block}\n\n"
            f"Response A:\n{response_a}\n\n"
            f"Response B:\n{response_b}"
        )

    async def _judge_with_openai(
        self,
        model: str,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        client = self._get_openai_client()
        response = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": self._judge_system_prompt()},
                {
                    "role": "user",
                    "content": self._single_judge_user_prompt(
                        question=question,
                        answer=answer,
                        ground_truth=ground_truth,
                    ),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_result",
                    "strict": True,
                    "schema": self._judge_schema(),
                },
            },
        )

        raw_text = response.choices[0].message.content or "{}"
        parsed = self._extract_json_from_text(raw_text)
        return self._normalize_single_judgment(parsed, model)

    async def _judge_with_claude(
        self,
        model: str,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> Dict[str, Any]:
        client = self._get_anthropic_client()
        response = await client.messages.create(
            model=model,
            max_tokens=900,
            temperature=0,
            system=self._judge_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": (
                        self._single_judge_user_prompt(
                            question=question,
                            answer=answer,
                            ground_truth=ground_truth,
                        )
                        + "\n\nReturn JSON only. No markdown."
                    ),
                }
            ],
        )

        raw_text = self._extract_anthropic_text(response)
        parsed = self._extract_json_from_text(raw_text)
        return self._normalize_single_judgment(parsed, model)

    async def _pairwise_with_openai(
        self,
        model: str,
        question: str,
        ground_truth: str,
        response_a: str,
        response_b: str,
    ) -> Dict[str, Any]:
        client = self._get_openai_client()
        response = await client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict and impartial evaluator.",
                },
                {
                    "role": "user",
                    "content": self._pairwise_user_prompt(
                        question=question,
                        ground_truth=ground_truth,
                        response_a=response_a,
                        response_b=response_b,
                    ),
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "pairwise_result",
                    "strict": True,
                    "schema": self._pairwise_schema(),
                },
            },
        )

        raw_text = response.choices[0].message.content or "{}"
        parsed = self._extract_json_from_text(raw_text)
        return self._normalize_pairwise_result(parsed)

    async def _pairwise_with_claude(
        self,
        model: str,
        question: str,
        ground_truth: str,
        response_a: str,
        response_b: str,
    ) -> Dict[str, Any]:
        client = self._get_anthropic_client()
        response = await client.messages.create(
            model=model,
            max_tokens=600,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": (
                        self._pairwise_user_prompt(
                            question=question,
                            ground_truth=ground_truth,
                            response_a=response_a,
                            response_b=response_b,
                        )
                        + "\n\nReturn JSON only. No markdown."
                    ),
                }
            ],
        )

        raw_text = self._extract_anthropic_text(response)
        parsed = self._extract_json_from_text(raw_text)
        return self._normalize_pairwise_result(parsed)

    def _extract_anthropic_text(self, response: Any) -> str:
        chunks = []
        for block in response.content:
            block_text = getattr(block, "text", None)
            if block_text:
                chunks.append(block_text)
        return "\n".join(chunks).strip()

    def _extract_json_from_text(self, text: Any) -> Dict[str, Any]:
        if not isinstance(text, str):
            text = str(text)

        trimmed = text.strip()
        if not trimmed:
            raise ValueError("Empty text response from Claude.")

        try:
            return json.loads(trimmed)
        except json.JSONDecodeError:
            pass

        fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", trimmed, flags=re.DOTALL)
        if fenced_match:
            return json.loads(fenced_match.group(1))

        candidate = self._find_first_json_object(trimmed)
        if not candidate:
            raise ValueError(f"Cannot extract JSON from Claude output: {trimmed}")
        return json.loads(candidate)

    def _find_first_json_object(self, text: str) -> Optional[str]:
        start = text.find("{")
        while start != -1:
            depth = 0
            in_string = False
            escape_next = False

            for index in range(start, len(text)):
                char = text[index]

                if escape_next:
                    escape_next = False
                    continue
                if char == "\\":
                    escape_next = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue

                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : index + 1]

            start = text.find("{", start + 1)
        return None

    def _normalize_single_judgment(self, payload: Dict[str, Any], model: str) -> Dict[str, Any]:
        raw_criteria = payload.get("criteria", {})
        criteria = {
            key: self._clamp_score(raw_criteria.get(key, 3))
            for key in self.CRITERIA
        }

        default_overall = sum(criteria.values()) / len(criteria)
        overall_score = payload.get("overall_score", default_overall)
        try:
            overall_score = float(overall_score)
        except (TypeError, ValueError):
            overall_score = default_overall
        overall_score = max(1.0, min(5.0, overall_score))

        reasoning = str(payload.get("reasoning", "")).strip() or f"{model} returned no explanation."

        return {
            "criteria": criteria,
            "overall_score": round(overall_score, 2),
            "reasoning": reasoning,
        }

    def _normalize_pairwise_result(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        winner = str(payload.get("winner", "Tie")).strip()
        winner = winner if winner in {"A", "B", "Tie"} else "Tie"

        confidence = self._clamp_score(payload.get("confidence", 3))
        reasoning = str(payload.get("reasoning", "")).strip() or "No reasoning provided."

        return {
            "winner": winner,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    def _calculate_agreement_rate(
        self,
        judgment_a: Dict[str, Any],
        judgment_b: Dict[str, Any],
    ) -> float:
        criterion_agreements = []
        for criterion in self.CRITERIA:
            diff = abs(judgment_a["criteria"][criterion] - judgment_b["criteria"][criterion])
            criterion_agreements.append(1 - (diff / 4))

        overall_diff = abs(judgment_a["overall_score"] - judgment_b["overall_score"])
        overall_agreement = 1 - (min(4.0, overall_diff) / 4)

        agreement_rate = (sum(criterion_agreements) + overall_agreement) / (len(criterion_agreements) + 1)
        return round(agreement_rate, 3)

    def _merge_two_judgments(
        self,
        judgment_a: Dict[str, Any],
        judgment_b: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged_criteria = {
            criterion: round(
                (judgment_a["criteria"][criterion] + judgment_b["criteria"][criterion]) / 2,
                2,
            )
            for criterion in self.CRITERIA
        }
        merged_overall = round(sum(merged_criteria.values()) / len(merged_criteria), 2)
        merged_reasoning = (
            "Consensus score from GPT-4o-mini and Claude Haiku. "
            f"GPT reason: {judgment_a['reasoning']} | Claude reason: {judgment_b['reasoning']}"
        )

        return {
            "criteria": merged_criteria,
            "overall_score": merged_overall,
            "reasoning": merged_reasoning,
        }

    def _is_position_biased(self, forward_winner: str, swapped_winner: str) -> bool:
        normalized_swapped = self._map_swapped_winner(swapped_winner)
        return forward_winner != normalized_swapped

    def _map_swapped_winner(self, swapped_winner: str) -> str:
        if swapped_winner == "A":
            return "B"
        if swapped_winner == "B":
            return "A"
        return "Tie"

    @staticmethod
    def _clamp_score(value: Any) -> int:
        try:
            numeric_value = int(round(float(value)))
        except (TypeError, ValueError):
            numeric_value = 3
        return max(1, min(5, numeric_value))
    

if __name__ == "__main__":
    async def test():
        judge = LLMJudge()
        result = await judge.evaluate_multi_judge(
            question="Chính sách bảo mật dữ liệu của công ty là gì?",
            answer="Công ty cam kết bảo vệ dữ liệu người dùng theo GDPR.",
            ground_truth="Công ty áp dụng mã hóa AES-256 và tuân thủ GDPR và ISO 27001.",
            fast_mode=False,
        )
        print(f"Score: {result['final_score']}")
        print(f"Agreement: {result['agreement_rate']}")
        print(f"Fast mode: {result.get('fast_mode')}")
        print(f"Fast mode short-circuit: {result.get('fast_mode_short_circuit')}")
        print(f"Conflict resolved: {result.get('conflict_resolved_by', result.get('conflict_resolved'))}")
        cost = result.get('cost_usd')
        print(f"Cost: ${cost}" if cost is not None else "Cost: N/A")
        print(f"Individual Scores: {result.get('individual_scores')}")
    
    asyncio.run(test())
