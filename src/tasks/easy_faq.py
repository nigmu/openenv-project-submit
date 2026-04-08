from __future__ import annotations

import json
import os
from typing import Any, Optional

from models import Action, Observation
from src.customers.scenarios import CustomerProfile, get_scenario
from src.tasks.base import BaseTask, TaskResult
from src.graders.base import BaseGrader
from src.utils.scoring import (
    analyze_sentiment,
    assess_conciseness,
    assess_professionalism,
    check_keyword_coverage,
    detect_hallucination,
    detect_intent,
)


def _load_faqs() -> dict:
    base = os.path.dirname(__file__)
    path = os.path.join(base, "..", "knowledge_base", "faqs.json")
    with open(path, "r") as f:
        return json.load(f)


class FAQTask(BaseTask):
    def __init__(self, scenario_index: int = 0, seed: Optional[int] = None):
        super().__init__(difficulty="easy", seed=seed)
        self.profile: CustomerProfile = get_scenario("easy", scenario_index)
        self.faqs = _load_faqs()
        self._scenario_key = self.profile.scenario_id.replace("faq_", "")

    def get_initial_observation(self) -> Observation:
        return self.build_observation(
            customer_message=self.profile.initial_message,
            sentiment=0.2,
            intent=detect_intent(self.profile.initial_message),
            urgency=self.profile.urgency,
            scenario_description=self.profile.scenario_description,
        )

    def evaluate_step(self, action: Action, customer_response: str) -> TaskResult:
        grader = FAQGrader()
        breakdown = grader.grade_step(action, customer_response, {
            "profile": self.profile,
            "faqs": self.faqs,
            "scenario_key": self._scenario_key,
        })

        score = sum(breakdown.values()) / len(breakdown)
        score = grader.normalize_score(score)

        feedback_parts = []
        if breakdown["correctness"] >= 0.8:
            feedback_parts.append("Accurate information provided")
        elif breakdown["correctness"] < 0.4:
            feedback_parts.append("Information may be inaccurate")
        if breakdown["professionalism"] >= 0.7:
            feedback_parts.append("Professional tone maintained")
        if breakdown["conciseness"] < 0.5:
            feedback_parts.append("Response could be more concise")
        if breakdown["hallucination"] < 0.5:
            feedback_parts.append("Possible hallucination detected")

        feedback = "; ".join(feedback_parts) if feedback_parts else "Response evaluated"

        return TaskResult(
            score=score,
            breakdown=breakdown,
            feedback=feedback,
            is_complete=score >= 0.7,
        )


class FAQGrader(BaseGrader):
    def grade_step(
        self,
        action: Action,
        customer_response: str,
        context: dict[str, Any],
    ) -> dict[str, float]:
        profile: CustomerProfile = context["profile"]
        faqs: dict = context["faqs"]
        scenario_key: str = context["scenario_key"]

        correctness = self._grade_correctness(action.message, profile, faqs, scenario_key)
        professionalism = assess_professionalism(action.message)
        conciseness = assess_conciseness(action.message, optimal_range=(30, 250))
        hallucination = detect_hallucination(action.message, profile.key_facts)

        return {
            "correctness": self.normalize_score(correctness),
            "professionalism": self.normalize_score(professionalism),
            "conciseness": self.normalize_score(conciseness),
            "hallucination": self.normalize_score(hallucination),
        }

    def _grade_correctness(
        self,
        response: str,
        profile: CustomerProfile,
        faqs: dict,
        scenario_key: str,
    ) -> float:
        keyword_score = check_keyword_coverage(response, profile.key_facts)

        faq_match_score = 0.0
        for faq_id, faq_data in faqs.items():
            if scenario_key in faq_id or any(kw in faq_id for kw in scenario_key.split("_")):
                faq_answer = faq_data["answer"]
                faq_keywords = faq_data.get("keywords", [])
                kw_coverage = check_keyword_coverage(response, faq_keywords)
                faq_match_score = max(faq_match_score, kw_coverage)

        return (keyword_score * 0.6) + (faq_match_score * 0.4)

    def grade_episode(
        self,
        action_history: list[Action],
        conversation_history: list[str],
        final_state: dict[str, Any],
    ) -> dict[str, float]:
        if not action_history:
            return {"final_score": 0.0}

        total_score = 0.0
        for action in action_history:
            step_scores = self.grade_step(action, "", {
                "profile": final_state.get("profile"),
                "faqs": final_state.get("faqs", {}),
                "scenario_key": final_state.get("scenario_key", ""),
            })
            total_score += sum(step_scores.values()) / len(step_scores)

        avg_score = total_score / len(action_history)

        if final_state.get("is_resolved"):
            avg_score = min(1.0, avg_score + 0.1)

        return {"final_score": self.normalize_score(avg_score)}
