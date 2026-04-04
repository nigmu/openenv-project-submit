"""Client for the Math Word Problem environment."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from demo.models import MathAction, MathObservation
except ModuleNotFoundError:
    from models import MathAction, MathObservation


class MathEnv(EnvClient[MathAction, MathObservation, State]):
    """
    Client for the Math Word Problem environment.

    This client uses a WebSocket session to the environment server. It is
    async-first; use ``.sync()`` for synchronous ``with`` / blocking code.

    Example (async):
        >>> async with MathEnv(base_url="http://localhost:8000") as env:
        ...     r = await env.reset(task_level="easy")
        ...     r = await env.step(MathAction(answer=42.0, reasoning="..."))

    Example (sync):
        >>> with MathEnv(base_url="http://localhost:8000").sync() as env:
        ...     r = env.reset(task_level="medium")
        ...     r = env.step(MathAction(answer=12.0))
    """

    def _step_payload(self, action: MathAction) -> Dict[str, Any]:
        return {
            "answer": action.answer,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[MathObservation]:
        obs_data = payload.get("observation", {})
        observation = MathObservation(
            problem=obs_data.get("problem", ""),
            task_level=obs_data.get("task_level", "easy"),
            correct_answer=obs_data.get("correct_answer"),
            is_correct=obs_data.get("is_correct", False),
            feedback=obs_data.get("feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata") or {},
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
