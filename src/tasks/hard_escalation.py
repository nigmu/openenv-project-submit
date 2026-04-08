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


class EscalationTask(BaseTask):
    def __init__(self, scenario_index: int = 0, seed: Optional[int] = None):
        super().__init__(difficulty="hard", seed=seed)
        self.profile: CustomerProfile = get_scenario("hard", scenario_index)
        self.procedures = _load_procedures()
        self._procedure_key = self._map_to_procedure()
        self._escalation_triggered = False
        self._de_escalation_attempts = 0

    def _map_to_procedure(self) -> str:
        issue = self.profile.issue_type
        if "angry" in issue or "defective" in issue:
            return "angry_customer"
        elif "late" in issue:
            return "late_delivery"
        return "angry_customer"

    def get_initial_observation(self) -> Observation:
        return self.build_observation(
            customer_message=self.profile.initial_message,
            sentiment=-0.8,
            intent=detect_intent(self.profile.initial_message),
            urgency=self.profile.urgency,
            scenario_description=self.profile.scenario_description,
        )

    def evaluate_step(self, action: Action, customer_response: str) -> TaskResult:
        grader = EscalationGrader()

        if action.action_type.value == "escalate":
            self._escalation_triggered = True
        if assess_empathy(action.message) > 0.5:
            self._de_escalation_attempts += 1

        breakdown = grader.grade_step(action, customer_response, {
            "profile": self.profile,
            "procedures": self.procedures,
            "procedure_key": self._procedure_key,
            "turn_count": self._turn_count,
            "conversation_history": self._conversation_history,
            "escalation_triggered": self._escalation_triggered,
            "de_escalation_attempts": self._de_escalation_attempts,
        })

        score = sum(breakdown.values()) / len(breakdown)
        score = grader.normalize_score(score)

        feedback_parts = []
        if breakdown["de_escalation"] >= 0.7:
            feedback_parts.append("Effective de-escalation")
        elif breakdown["de_escalation"] < 0.3:
            feedback_parts.append("De-escalation ineffective")
        if breakdown["protocol_compliance"] >= 0.7:
            feedback_parts.append("Good protocol compliance")
        elif breakdown["protocol_compliance"] < 0.3:
            feedback_parts.append("Protocol compliance poor")
        if breakdown["resolution_path"] >= 0.7:
            feedback_parts.append("Clear resolution path")

        feedback = "; ".join(feedback_parts) if feedback_parts else "Response evaluated"

        return TaskResult(
            score=score,
            breakdown=breakdown,
            feedback=feedback,
            is_complete=score >= 0.8,
        )


class EscalationGrader(BaseGrader):
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
        escalation_triggered: bool = context.get("escalation_triggered", False)
        de_escalation_attempts: int = context.get("de_escalation_attempts", 0)

        de_escalation = self._grade_de_escalation(action.message, action.action_type, turn_count, de_escalation_attempts)
        info_gathering = self._grade_info_gathering(action.message, profile, conversation_history)
        resolution_path = self._grade_resolution_path(action.message, action.action_type, procedures, procedure_key, escalation_triggered, profile)
        satisfaction_trajectory = self._grade_satisfaction_trajectory(conversation_history, turn_count)
        protocol_compliance = self._grade_protocol_compliance(action.message, action.action_type, procedures, procedure_key, escalation_triggered, profile)

        return {
            "de_escalation": self.normalize_score(de_escalation),
            "information_gathering": self.normalize_score(info_gathering),
            "resolution_path": self.normalize_score(resolution_path),
            "satisfaction_trajectory": self.normalize_score(satisfaction_trajectory),
            "protocol_compliance": self.normalize_score(protocol_compliance),
        }

    def _grade_de_escalation(
        self,
        message: str,
        action_type,
        turn_count: int,
        de_escalation_attempts: int,
    ) -> float:
        score = 0.0

        empathy_phrases = [
            "i understand", "i'm sorry", "i apologize", "frustrating",
            "inconvenience", "upsetting", "disappointing", "i can imagine",
            "that must be", "we value", "your concern", "we take seriously",
        ]
        empathy_count = sum(1 for phrase in empathy_phrases if phrase in message.lower())
        score += min(0.4, empathy_count * 0.15)

        if action_type.value in ["acknowledge", "ask_clarify"]:
            score += 0.15

        if action_type.value == "escalate" and turn_count >= 2:
            score += 0.2
        elif action_type.value == "escalate" and turn_count < 2:
            score -= 0.2

        if de_escalation_attempts >= 2:
            score += 0.15

        calming_words = ["calm", "help", "resolve", "fix", "solution", "work together", "sort this out"]
        calming_count = sum(1 for word in calming_words if word in message.lower())
        score += min(0.15, calming_count * 0.05)

        return max(0.0, min(1.0, score))

    def _grade_info_gathering(
        self,
        message: str,
        profile: CustomerProfile,
        conversation_history: list[str],
    ) -> float:
        if not conversation_history:
            return 0.3

        key_facts_mentioned = check_keyword_coverage(message, profile.key_facts)

        question_count = message.count("?")
        question_score = min(0.3, question_count * 0.1)

        context_awareness = 0.0
        if len(conversation_history) > 1:
            context_awareness = 0.2

        return min(1.0, key_facts_mentioned * 0.5 + question_score + context_awareness)

    def _grade_resolution_path(
        self,
        message: str,
        action_type,
        procedures: dict,
        procedure_key: str,
        escalation_triggered: bool,
        profile: CustomerProfile,
    ) -> float:
        score = 0.0

        procedure = procedures.get(procedure_key, {})
        escalation_threshold = procedure.get("escalation_threshold", "")

        wants_manager = any(trigger in profile.initial_message.lower() for trigger in ["manager", "supervisor"])

        if wants_manager and escalation_triggered:
            score += 0.5
        elif wants_manager and not escalation_triggered:
            score += 0.1

        resolution_actions = {
            "refund": ["refund", "money back", "credit back"],
            "replacement": ["replacement", "replace", "new one"],
            "compensation": ["compensation", "compensate", "discount", "credit"],
            "investigate": ["investigate", "look into", "review", "check"],
        }

        actions_mentioned = sum(
            1 for words in resolution_actions.values()
            if any(word in message.lower() for word in words)
        )
        score += min(0.3, actions_mentioned * 0.1)

        if action_type.value == "close" and score < 0.5:
            score *= 0.5

        return min(1.0, score)

    def _grade_satisfaction_trajectory(
        self,
        conversation_history: list[str],
        turn_count: int,
    ) -> float:
        if len(conversation_history) < 2:
            return 0.5

        positive_signals = ["thank", "appreciate", "helpful", "good", "okay", "understood", "great"]
        negative_signals = ["unacceptable", "terrible", "worst", "angry", "frustrated", "ridiculous", "manager"]

        recent_messages = conversation_history[-2:]
        positive_count = sum(
            1 for msg in recent_messages
            for signal in positive_signals
            if signal in msg.lower()
        )
        negative_count = sum(
            1 for msg in recent_messages
            for signal in negative_signals
            if signal in msg.lower()
        )

        if positive_count > negative_count:
            return 0.8
        elif positive_count == negative_count:
            return 0.5
        else:
            return 0.3

    def _grade_protocol_compliance(
        self,
        message: str,
        action_type,
        procedures: dict,
        procedure_key: str,
        escalation_triggered: bool,
        profile: CustomerProfile,
    ) -> float:
        score = 0.0

        procedure = procedures.get(procedure_key, {})
        steps = procedure.get("steps", [])

        if steps:
            step_keywords_map = {
                "angry_customer": {
                    "remain_calm": ["professional", "understand", "help", "resolve"],
                    "acknowledge": ["frustration", "concern", "experience", "situation"],
                    "listen": ["hear", "listening", "understand", "concern"],
                    "apologize": ["sorry", "apologize", "apologies", "regret"],
                    "solutions": ["solution", "resolve", "fix", "help", "make right"],
                    "supervisor": ["supervisor", "manager", "escalate", "senior"],
                },
            }

            keywords = step_keywords_map.get(procedure_key, {})
            if keywords:
                matched = sum(
                    1 for step_name, words in keywords.items()
                    if any(word in message.lower() for word in words)
                )
                score += (matched / len(keywords)) * 0.6

        if action_type.value == "escalate" and escalation_triggered:
            score += 0.3

        if action_type.value == "close" and score < 0.6:
            score *= 0.7

        return min(1.0, score)

    def grade_episode(
        self,
        action_history: list[Action],
        conversation_history: list[str],
        final_state: dict[str, Any],
    ) -> dict[str, float]:
        if not action_history:
            return {"final_score": 0.0}

        total_scores = {
            "de_escalation": 0.0,
            "information_gathering": 0.0,
            "resolution_path": 0.0,
            "satisfaction_trajectory": 0.0,
            "protocol_compliance": 0.0,
        }

        escalation_triggered = final_state.get("escalation_triggered", False)
        de_escalation_attempts = final_state.get("de_escalation_attempts", 0)

        for i, action in enumerate(action_history):
            step_scores = self.grade_step(action, "", {
                "profile": final_state.get("profile"),
                "procedures": final_state.get("procedures", {}),
                "procedure_key": final_state.get("procedure_key", ""),
                "turn_count": i + 1,
                "conversation_history": conversation_history[:i+1],
                "escalation_triggered": escalation_triggered,
                "de_escalation_attempts": de_escalation_attempts,
            })
            for key in total_scores:
                total_scores[key] += step_scores.get(key, 0.0)

        avg_scores = {k: v / len(action_history) for k, v in total_scores.items()}

        if final_state.get("is_resolved"):
            avg_scores["resolution_path"] = min(1.0, avg_scores["resolution_path"] + 0.2)

        final_score = sum(avg_scores.values()) / len(avg_scores)
        return {"final_score": self.normalize_score(final_score)}
