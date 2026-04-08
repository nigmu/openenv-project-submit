from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .scenarios import CustomerProfile


@dataclass
class CustomerState:
    profile: CustomerProfile
    current_mood: float
    satisfaction_history: list[float] = field(default_factory=list)
    conversation_turns: int = 0
    is_escalated: bool = False
    is_resolved: bool = False
    frustration_level: float = 0.0
    last_response_quality: float = 0.5
    has_been_addressed: bool = False


class CustomerSimulator:
    def __init__(self, profile: CustomerProfile, seed: Optional[int] = None):
        self.profile = profile
        self.state = CustomerState(
            profile=profile,
            current_mood=profile.mood,
        )
        self._rng = random.Random(seed)
        self._response_templates = self._build_response_templates()

    def _build_response_templates(self) -> dict:
        return {
            "positive_ack": [
                "Thank you, that's helpful!",
                "Okay, I appreciate the information.",
                "Great, thanks for letting me know.",
                "That makes sense, thank you.",
                "Perfect, that's exactly what I needed to know.",
                "Got it, thank you for the clear explanation.",
            ],
            "neutral_ack": [
                "Okay.",
                "I see.",
                "Alright.",
                "Understood.",
            ],
            "negative_ack": [
                "That's not quite what I was asking.",
                "I don't think you understood my question.",
                "That doesn't really help me.",
                "I'm not satisfied with that answer.",
            ],
            "frustrated": [
                "This is really frustrating.",
                "I've already explained this to you.",
                "Are you even listening to me?",
                "This is not acceptable.",
            ],
            "angry": [
                "This is unacceptable!",
                "I want to speak to a manager!",
                "I'm extremely disappointed with this service!",
                "This is the worst customer experience I've ever had!",
            ],
            "follow_up": [
                "Can you tell me more about that?",
                "What about the other thing I mentioned?",
                "Is there anything else I should know?",
                "Can you clarify that point?",
            ],
            "resolution_check": [
                "So just to confirm, that should resolve my issue?",
                "Will that actually fix the problem?",
                "How long will that take to process?",
                "Can you send me a confirmation of this?",
            ],
            "grateful": [
                "Thank you so much for your help!",
                "I really appreciate you sorting this out for me.",
                "That's great, thanks for taking care of it!",
                "Wonderful, thank you for the quick resolution!",
            ],
        }

    def get_initial_message(self) -> str:
        return self.profile.initial_message

    def generate_response(self, agent_message: str, action_type: str, response_quality: float) -> str:
        self.state.conversation_turns += 1
        self.state.last_response_quality = response_quality

        mood_delta = self._calculate_mood_delta(agent_message, action_type, response_quality)
        self.state.current_mood = max(0.0, min(10.0, self.state.current_mood + mood_delta))
        self.state.satisfaction_history.append(self.state.current_mood)

        if response_quality < 0.3:
            self.state.frustration_level = min(1.0, self.state.frustration_level + 0.25)
        elif response_quality > 0.7:
            self.state.frustration_level = max(0.0, self.state.frustration_level - 0.15)

        if response_quality >= 0.6:
            self.state.has_been_addressed = True

        if self.state.frustration_level > 0.7 and not self.state.is_escalated:
            if self._rng.random() < 0.4:
                self.state.is_escalated = True

        return self._generate_contextual_response(response_quality, action_type)

    def _calculate_mood_delta(self, agent_message: str, action_type: str, quality: float) -> float:
        base_delta = (quality - 0.5) * 2.0

        if action_type == "acknowledge" and quality > 0.6:
            base_delta += 0.3
        elif action_type == "ask_clarify":
            base_delta += 0.1
        elif action_type == "escalate" and self.state.frustration_level > 0.5:
            base_delta += 0.5
        elif action_type == "close" and self.state.current_mood < 4.0:
            base_delta -= 1.5

        if len(agent_message) > 500:
            base_delta -= 0.2
        elif len(agent_message) < 10:
            base_delta -= 0.3

        return base_delta * (self.profile.patience / 10.0)

    def _generate_contextual_response(self, quality: float, action_type: str) -> str:
        mood = self.state.current_mood

        if quality >= 0.7:
            if mood >= 6.0:
                if self.state.has_been_addressed and self.state.conversation_turns >= 2:
                    return self._rng.choice(self._response_templates["grateful"])
                if self.state.conversation_turns >= 3:
                    return self._rng.choice(self._response_templates["resolution_check"])
                return self._rng.choice(self._response_templates["positive_ack"])
            elif mood >= 4.0:
                return self._rng.choice(self._response_templates["positive_ack"])
            else:
                return self._rng.choice(self._response_templates["follow_up"])

        elif quality >= 0.4:
            if mood >= 7.0:
                return self._rng.choice(self._response_templates["positive_ack"])
            elif mood >= 4.0:
                if self.state.conversation_turns <= 1:
                    return self._rng.choice(self._response_templates["follow_up"])
                return self._rng.choice(self._response_templates["neutral_ack"])
            else:
                if self.state.conversation_turns <= 2:
                    return self._rng.choice(self._response_templates["follow_up"])
                return self._rng.choice(self._response_templates["negative_ack"])

        else:
            if self.state.is_escalated:
                return self._rng.choice(self._response_templates["angry"])
            elif self.state.conversation_turns <= 1:
                return self._rng.choice(self._response_templates["negative_ack"])
            elif mood < 3.0:
                return self._rng.choice(self._response_templates["frustrated"])
            else:
                return self._rng.choice(self._response_templates["negative_ack"])

    def is_done(self, max_turns: int = 8) -> bool:
        if self.state.conversation_turns >= max_turns:
            return True
        if self.state.is_resolved:
            return True
        if self.state.current_mood <= 0.5 and self.state.conversation_turns >= 3:
            return True
        return False

    def mark_resolved(self):
        self.state.is_resolved = True
        self.state.current_mood = min(10.0, self.state.current_mood + 2.0)
        self.state.satisfaction_history.append(self.state.current_mood)
