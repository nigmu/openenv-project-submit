from __future__ import annotations

import random
from typing import Any, Optional

from models import Action, Observation, Reward, State, ActionType
from src.customers.scenarios import get_scenario, CustomerProfile
from src.customers.simulator import CustomerSimulator
from src.tasks.base import BaseTask, TaskResult
from src.tasks.easy_faq import FAQTask
from src.tasks.medium_complaint import ComplaintTask
from src.tasks.hard_escalation import EscalationTask
from src.utils.scoring import analyze_sentiment, detect_intent
from src.utils.validators import validate_action


class CustomerServiceEnv:
    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._rng = random.Random(seed)

        self._task: Optional[BaseTask] = None
        self._customer: Optional[CustomerSimulator] = None
        self._profile: Optional[CustomerProfile] = None

        self._current_observation: Optional[Observation] = None
        self._episode_done: bool = False
        self._turn_count: int = 0
        self._action_history: list[Action] = []
        self._conversation_history: list[str] = []
        self._satisfaction_trajectory: list[float] = []
        self._total_reward: float = 0.0

        self._max_turns: int = 10
        self._task_type: str = "easy"

    def reset(self, task_type: Optional[str] = None, scenario_index: int = 0) -> Observation:
        self._task_type = task_type or self._rng.choice(["easy", "medium", "hard"])
        self._task = self._create_task(self._task_type, scenario_index)
        self._profile = self._task.profile

        self._customer = CustomerSimulator(self._profile, seed=self._seed)
        self._max_turns = self._task.max_turns

        self._episode_done = False
        self._turn_count = 0
        self._action_history = []
        self._conversation_history = []
        self._satisfaction_trajectory = []
        self._total_reward = 0.0

        self._current_observation = self._task.get_initial_observation()
        self._conversation_history.append(self._profile.initial_message)
        self._satisfaction_trajectory.append(self._profile.mood)

        return self._current_observation

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict[str, Any]]:
        if self._episode_done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        validation = validate_action(action)
        if not validation["valid"]:
            penalty_reward = Reward(
                value=0.0,
                breakdown={"validation_penalty": 0.0},
                feedback=f"Invalid action: {', '.join(validation['issues'])}",
            )
            return self._current_observation, penalty_reward, True, {"validation_errors": validation["issues"]}

        self._action_history.append(action)
        self._turn_count += 1

        task_result = self._task.evaluate_step(action, "")

        response_quality = task_result.score
        customer_response = self._customer.generate_response(action.message, action.action_type.value, response_quality)

        self._conversation_history.append(customer_response)
        self._satisfaction_trajectory.append(self._customer.state.current_mood)

        sentiment = analyze_sentiment(customer_response)
        intent = detect_intent(customer_response)

        step_reward = self._calculate_step_reward(action, task_result, customer_response, sentiment)
        self._total_reward += step_reward.value

        self._current_observation = Observation(
            customer_message=customer_response,
            sentiment=sentiment,
            intent=intent,
            urgency=self._profile.urgency,
            conversation_history=self._conversation_history.copy(),
            turn_count=self._turn_count,
            task_type=self._task_type,
            scenario_description=self._profile.scenario_description,
        )

        self._episode_done = self._check_episode_done(task_result, customer_response)

        if self._episode_done and self._customer.state.is_resolved:
            step_reward.value = min(1.0, step_reward.value + 0.2)
            step_reward.feedback += "; Episode resolved successfully"

        info = {
            "turn_count": self._turn_count,
            "customer_mood": self._customer.state.current_mood,
            "is_resolved": self._customer.state.is_resolved,
            "is_escalated": self._customer.state.is_escalated,
            "satisfaction_trajectory": self._satisfaction_trajectory.copy(),
            "task_result": {
                "score": task_result.score,
                "breakdown": task_result.breakdown,
                "feedback": task_result.feedback,
            },
        }

        return self._current_observation, step_reward, self._episode_done, info

    def state(self) -> State:
        return State(
            customer_mood=self._customer.state.current_mood if self._customer else 5.0,
            satisfaction_trajectory=self._satisfaction_trajectory.copy(),
            resolution_status="resolved" if (self._customer and self._customer.state.is_resolved) else "ongoing",
            turn_count=self._turn_count,
            task_type=self._task_type,
            conversation_history=self._conversation_history.copy(),
            episode_done=self._episode_done,
        )

    def _create_task(self, task_type: str, scenario_index: int) -> BaseTask:
        if task_type == "easy":
            return FAQTask(scenario_index=scenario_index, seed=self._seed)
        elif task_type == "medium":
            return ComplaintTask(scenario_index=scenario_index, seed=self._seed)
        elif task_type == "hard":
            return EscalationTask(scenario_index=scenario_index, seed=self._seed)
        else:
            raise ValueError(f"Unknown task type: {task_type}. Must be 'easy', 'medium', or 'hard'.")

    def _calculate_step_reward(
        self,
        action: Action,
        task_result: TaskResult,
        customer_response: str,
        sentiment: float,
    ) -> Reward:
        base_score = task_result.score

        sentiment_delta = 0.0
        if len(self._satisfaction_trajectory) > 1:
            prev_mood = self._satisfaction_trajectory[-2]
            curr_mood = self._customer.state.current_mood
            sentiment_delta = (curr_mood - prev_mood) / 10.0

        progress_reward = base_score * 0.4
        sentiment_reward = max(-0.2, min(0.2, sentiment_delta * 0.3))
        action_bonus = self._action_type_bonus(action)
        verbosity_penalty = -0.1 if len(action.message) > 500 else 0.0

        step_value = progress_reward + sentiment_reward + action_bonus + verbosity_penalty
        step_value = max(0.0, min(1.0, step_value))

        feedback = task_result.feedback
        if sentiment_delta > 0.2:
            feedback += "; Customer sentiment improving"
        elif sentiment_delta < -0.2:
            feedback += "; Customer sentiment worsening"

        return Reward(
            value=step_value,
            breakdown={
                "progress": progress_reward,
                "sentiment_delta": sentiment_reward,
                "action_bonus": action_bonus,
                "verbosity_penalty": verbosity_penalty,
            },
            feedback=feedback,
        )

    def _action_type_bonus(self, action: Action) -> float:
        if action.action_type == ActionType.ACKNOWLEDGE:
            return 0.05
        elif action.action_type == ActionType.ASK_CLARIFY:
            return 0.03
        elif action.action_type == ActionType.ESCALATE:
            if self._customer and self._customer.state.frustration_level > 0.5:
                return 0.1
            return -0.05
        elif action.action_type == ActionType.CLOSE:
            if self._customer and self._customer.state.current_mood >= 6.0:
                return 0.1
            return -0.1
        return 0.0

    def _check_episode_done(self, task_result: TaskResult, customer_response: str) -> bool:
        if self._turn_count >= self._max_turns:
            return True

        if self._customer and self._customer.state.is_resolved:
            return True

        if self._customer and self._customer.state.current_mood <= 0.5 and self._turn_count >= 3:
            return True

        if task_result.is_complete and self._turn_count >= 2:
            return True

        return False
