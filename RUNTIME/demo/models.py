"""
Data models for the Math Word Problem Environment.

The agent receives a math problem and must return a numerical answer.
Reward is based on how close the answer is to the correct one.
"""

from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class MathAction(Action):
    """Action: the agent submits a numerical answer and optional reasoning."""

    answer: float = Field(..., description="Numerical answer to the math problem")
    reasoning: str = Field(
        default="",
        description="Step-by-step reasoning (optional, not graded)"
    )


class MathObservation(Observation):
    """Observation: the problem to solve, plus feedback after a step."""

    problem: str = Field(
        default="",
        description="The math word problem the agent must solve"
    )
    task_level: str = Field(
        default="easy",
        description="Difficulty: easy | medium | hard"
    )
    correct_answer: Optional[float] = Field(
        default=None,
        description="The correct answer — revealed only after step(), not at reset()"
    )
    is_correct: bool = Field(
        default=False,
        description="True if the submitted answer was exactly correct"
    )
    feedback: str = Field(
        default="",
        description="Human-readable feedback on the submitted answer"
    )   