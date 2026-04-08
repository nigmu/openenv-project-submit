from __future__ import annotations

from typing import Any

from models import Action


def validate_action(action: Action) -> dict[str, str]:
    issues = []

    if not action.message or len(action.message.strip()) == 0:
        issues.append("message_empty")

    if len(action.message) > 2000:
        issues.append("message_too_long")

    if action.confidence < 0.0 or action.confidence > 1.0:
        issues.append("confidence_out_of_range")

    return {"valid": len(issues) == 0, "issues": issues}
