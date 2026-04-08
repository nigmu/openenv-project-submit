from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from models import Action, Observation, Reward


@dataclass
class TaskResult:
    score: float
    breakdown: dict[str, float]
    feedback: str
    is_complete: bool = False


class BaseTask(ABC):
    def __init__(self, difficulty: str, seed: Optional[int] = None):
        self.difficulty = difficulty
        self.seed = seed
        self.max_turns = self._get_max_turns()
        self._turn_count = 0
        self._conversation_history: list[str] = []
        self._action_history: list[Action] = []

    @abstractmethod
    def get_initial_observation(self) -> Observation:
        pass

    @abstractmethod
    def evaluate_step(self, action: Action, customer_response: str) -> TaskResult:
        pass

    def _get_max_turns(self) -> int:
        return {"easy": 5, "medium": 7, "hard": 10}.get(self.difficulty, 6)

    def record_step(self, action: Action, customer_message: str):
        self._turn_count += 1
        self._conversation_history.append(customer_message)
        self._action_history.append(action)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def conversation_history(self) -> list[str]:
        return self._conversation_history.copy()

    @property
    def is_done(self) -> bool:
        return self._turn_count >= self.max_turns

    def build_observation(
        self,
        customer_message: str,
        sentiment: float,
        intent: str,
        urgency: int,
        scenario_description: str,
    ) -> Observation:
        return Observation(
            customer_message=customer_message,
            sentiment=sentiment,
            intent=intent,
            urgency=urgency,
            conversation_history=self._conversation_history.copy(),
            turn_count=self._turn_count,
            task_type=self.difficulty,
            scenario_description=scenario_description,
        )
