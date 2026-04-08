from __future__ import annotations

import json
import os
from typing import Any, Optional

from models import Action, Observation
from src.customers.scenarios import CustomerProfile, get_scenario
from src.tasks.base import BaseTask, TaskResult
from src.graders.base import BaseGrader
from src.utils.scoring import (
    assess_conciseness,
    assess_empathy,
    assess_professionalism,
    check_keyword_coverage,
    detect_hallucination,
    detect_intent,
)


def _load_procedures() -> dict:
    base = os.path.dirname(__file__)
    path = os.path.join(base, "..", "knowledge_base", "procedures.json")
    with open(path, "r") as f:
        return json.load(f)


class ComplaintTask(BaseTask):
    def __init__(self, scenario_index: int = 0, seed: Optional[int] = None):
        super().__init__(difficulty="medium", seed=seed)
        self.profile: CustomerProfile = get_scenario("medium", scenario_index)
        self.procedures = _load_procedures()
        self._procedure_key = self._map_to_procedure()

    def _map_to_procedure(self) -> str:
        issue = self.profile.issue_type
        mapping = {
            "defective_product": "defective_product",
            "late_delivery": "late_delivery",
            "billing_dispute": "billing_dispute",
        }
        return mapping.get(issue, "defective_product")

    def get_initial_observation(self) -> Observation:
        return self.build_observation(
            customer_message=self.profile.initial_message,
            sentiment=-0.5,
            intent=detect_intent(self.profile.initial_message),
            urgency=self.profile.urgency,
            scenario_description=self.profile.scenario_description,
        )

    def evaluate_step(self, action: Action, customer_response: str) -> TaskResult:
        grader = ComplaintGrader()
        breakdown = grader.grade_step(action, customer_response, {
            "profile": self.profile,
            "procedures": self.procedures,
            "procedure_key": self._procedure_key,
            "turn_count": self._turn_count,
            "conversation_history": self._conversation_history,
        })

        score = sum(breakdown.values()) / len(breakdown)
        score = grader.normalize_score(score)

        feedback_parts = []
        if breakdown["empathy"] >= 0.7:
            feedback_parts.append("Good empathy shown")
        elif breakdown["empathy"] < 0.3:
            feedback_parts.append("Lack of empathy detected")
        if breakdown["solution"] >= 0.7:
            feedback_parts.append("Appropriate solution offered")
        elif breakdown["solution"] < 0.3:
            feedback_parts.append("Solution inadequate")
        if breakdown["professionalism"] >= 0.7:
            feedback_parts.append("Professional tone maintained")

        feedback = "; ".join(feedback_parts) if feedback_parts else "Response evaluated"

        return TaskResult(
            score=score,
            breakdown=breakdown,
            feedback=feedback,
            is_complete=score >= 0.75,
        )


class ComplaintGrader(BaseGrader):
    def grade_step(
        self,
        action: Action,
        customer_response: str,
        context: dict[str, Any],
    ) -> dict[str, float]:
        profile: CustomerProfile = context["profile"]
        procedures: dict = context["procedures"]
        procedure_key: str = context["procedure_key"]
        turn_count: int = context.get("turn_count", 1)
        conversation_history: list[str] = context.get("conversation_history", [])

        empathy = assess_empathy(action.message)
        problem_id = self._grade_problem_identification(action.message, profile)
        solution = self._grade_solution(action.message, procedures, procedure_key, turn_count)
        professionalism = assess_professionalism(action.message)

        return {
            "empathy": self.normalize_score(empathy),
            "problem_identification": self.normalize_score(problem_id),
            "solution": self.normalize_score(solution),
            "professionalism": self.normalize_score(professionalism),
        }

    def _grade_problem_identification(self, response: str, profile: CustomerProfile) -> float:
        return check_keyword_coverage(response, profile.key_facts)

    def _grade_solution(
        self,
        response: str,
        procedures: dict,
        procedure_key: str,
        turn_count: int,
    ) -> float:
        procedure = procedures.get(procedure_key, {})
        steps = procedure.get("steps", [])

        if not steps:
            return 0.5

        step_keywords = {
            "defective_product": {
                "apologize": ["sorry", "apologize", "apologies", "understand"],
                "replace": ["replacement", "replace", "new one", "send another"],
                "refund": ["refund", "money back", "full refund"],
                "return_label": ["return label", "return shipping", "prepaid"],
                "expedite": ["expedite", "express", "fast", "priority", "rush"],
            },
            "late_delivery": {
                "tracking": ["tracking", "look up", "check", "status"],
                "explain": ["delay", "carrier", "shipping", "transit"],
                "shipping_refund": ["shipping refund", "refund shipping", "compensate shipping"],
                "updated_estimate": ["updated", "estimate", "new delivery", "when arrive"],
            },
            "billing_dispute": {
                "verify": ["verify", "check", "look into", "review", "investigate"],
                "explain": ["explain", "charge", "transaction", "statement"],
                "refund": ["refund", "reverse", "credit back", "remove charge"],
                "reference": ["reference", "confirmation", "case number", "ticket"],
                "follow_up": ["follow up", "follow-up", "contact you", "update you"],
            },
        }

        keywords = step_keywords.get(procedure_key, {})

        if not keywords:
            return check_keyword_coverage(response, [s.lower() for s in steps[:2]])

        matched_steps = 0
        for step_name, words in keywords.items():
            if any(word in response.lower() for word in words):
                matched_steps += 1

        return matched_steps / len(keywords)

    def grade_episode(
        self,
        action_history: list[Action],
        conversation_history: list[str],
        final_state: dict[str, Any],
    ) -> dict[str, float]:
        if not action_history:
            return {"final_score": 0.0}

        total_scores = {"empathy": 0.0, "problem_identification": 0.0, "solution": 0.0, "professionalism": 0.0}

        for i, action in enumerate(action_history):
            step_scores = self.grade_step(action, "", {
                "profile": final_state.get("profile"),
                "procedures": final_state.get("procedures", {}),
                "procedure_key": final_state.get("procedure_key", ""),
                "turn_count": i + 1,
                "conversation_history": conversation_history[:i+1],
            })
            for key in total_scores:
                total_scores[key] += step_scores.get(key, 0.0)

        avg_scores = {k: v / len(action_history) for k, v in total_scores.items()}

        if final_state.get("is_resolved"):
            avg_scores["solution"] = min(1.0, avg_scores["solution"] + 0.15)

        final_score = sum(avg_scores.values()) / len(avg_scores)
        return {"final_score": self.normalize_score(final_score)}
