from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from models import Action


class BaseGrader(ABC):
    @abstractmethod
    def grade_step(self, action: Action, customer_response: str, context: dict[str, Any]) -> dict[str, float]:
        pass

    @abstractmethod
    def grade_episode(self, action_history: list[Action], conversation_history: list[str], final_state: dict[str, Any]) -> dict[str, float]:
        pass

    def normalize_score(self, raw_score: float) -> float:
        return max(0.0, min(1.0, raw_score))
