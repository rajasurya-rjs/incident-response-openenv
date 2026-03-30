"""Incident Response Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    IncidentResponseAction,
    IncidentResponseObservation,
    IncidentResponseState,
)


class IncidentResponseEnv(
    EnvClient[IncidentResponseAction, IncidentResponseObservation, IncidentResponseState]
):
    """Client for the Incident Response Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with IncidentResponseEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(task_id="disk_full")
        ...     print(result.observation.alert_summary)
        ...
        ...     result = env.step(IncidentResponseAction(
        ...         command="check_metrics",
        ...         target="postgres-primary",
        ...     ))
        ...     print(result.observation.command_output)
    """

    def _step_payload(self, action: IncidentResponseAction) -> Dict:
        """Convert IncidentResponseAction to JSON payload for step message."""
        payload = {
            "command": action.command,
            "target": action.target,
        }
        if action.parameters:
            payload["parameters"] = action.parameters
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[IncidentResponseObservation]:
        """Parse server response into StepResult[IncidentResponseObservation]."""
        obs_data = payload.get("observation", {})
        observation = IncidentResponseObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            alert_summary=obs_data.get("alert_summary", ""),
            command_output=obs_data.get("command_output", ""),
            available_commands=obs_data.get("available_commands", []),
            available_services=obs_data.get("available_services", []),
            services_investigated=obs_data.get("services_investigated", []),
            actions_taken=obs_data.get("actions_taken", []),
            time_elapsed_minutes=obs_data.get("time_elapsed_minutes", 0),
            severity=obs_data.get("severity", "warning"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> IncidentResponseState:
        """Parse server response into IncidentResponseState."""
        return IncidentResponseState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            difficulty=payload.get("difficulty", "easy"),
            max_steps=payload.get("max_steps", 20),
        )
