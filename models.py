from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    ANSWER = "answer"
    ACKNOWLEDGE = "acknowledge"
    ASK_CLARIFY = "ask_clarify"
    ESCALATE = "escalate"
    CLOSE = "close"


class Action(BaseModel):
    message: str = Field(
        description="The response message to send to the customer"
    )
    action_type: ActionType = Field(
        description="The type of action being taken"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Agent confidence in this response (0.0-1.0)",
    )


class Observation(BaseModel):
    customer_message: str = Field(
        description="The customer's latest message"
    )
    sentiment: float = Field(
        description="Current customer sentiment (-1.0 negative to 1.0 positive)"
    )
    intent: str = Field(
        description="Detected customer intent category"
    )
    urgency: int = Field(
        ge=1,
        le=5,
        description="Urgency level (1=low, 5=critical)",
    )
    conversation_history: list[str] = Field(
        default_factory=list,
        description="Full conversation transcript",
    )
    turn_count: int = Field(
        description="Current turn number"
    )
    task_type: str = Field(
        description="Current task difficulty (easy/medium/hard)"
    )
    scenario_description: str = Field(
        default="",
        description="Brief description of the scenario for context",
    )


class Reward(BaseModel):
    value: float = Field(
        ge=0.0,
        le=1.0,
        description="Reward score for this step (0.0-1.0)",
    )
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Detailed scoring breakdown",
    )
    feedback: str = Field(
        default="",
        description="Human-readable feedback about this step",
    )


class State(BaseModel):
    customer_mood: float = Field(
        ge=0.0,
        le=10.0,
        description="Customer mood (0=angry, 10=delighted)",
    )
    satisfaction_trajectory: list[float] = Field(
        default_factory=list,
        description="Satisfaction score after each turn",
    )
    resolution_status: str = Field(
        description="Current resolution status",
    )
    turn_count: int = Field(
        description="Current turn number"
    )
    task_type: str = Field(
        description="Current task difficulty"
    )
    conversation_history: list[str] = Field(
        default_factory=list,
        description="Full conversation transcript",
    )
    episode_done: bool = Field(
        description="Whether the episode has ended"
    )
