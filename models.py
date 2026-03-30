"""Data models for the Incident Response Environment."""

from typing import Any, Dict, List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class IncidentResponseAction(Action):
    """Agent's action in the incident response environment.

    The agent can investigate services (check_metrics, read_logs, check_status),
    take remediation actions (restart_service, scale_service, rollback_deploy, clear_disk),
    submit a diagnosis, or escalate to a senior engineer.
    """

    command: str = Field(
        ...,
        description=(
            "One of: check_metrics, read_logs, check_status, "
            "restart_service, scale_service, rollback_deploy, "
            "clear_disk, diagnose, escalate"
        ),
    )
    target: str = Field(
        default="",
        description="Target service name (e.g., 'api-gateway', 'postgres-primary')",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters (e.g., {'explanation': '...'} for diagnose, {'replicas': 4} for scale_service)",
    )


class IncidentResponseObservation(Observation):
    """What the agent observes after each action.

    Inherits done (bool), reward (float|None), and metadata (dict) from Observation.
    """

    alert_summary: str = Field(
        default="",
        description="Current alert text and severity",
    )
    command_output: str = Field(
        default="",
        description="Output from the last command executed",
    )
    available_commands: List[str] = Field(
        default_factory=list,
        description="List of valid commands the agent can issue",
    )
    available_services: List[str] = Field(
        default_factory=list,
        description="List of valid service targets",
    )
    services_investigated: List[str] = Field(
        default_factory=list,
        description="Services the agent has already investigated",
    )
    actions_taken: List[str] = Field(
        default_factory=list,
        description="History of actions taken so far",
    )
    time_elapsed_minutes: int = Field(
        default=0,
        description="Simulated minutes elapsed since alert fired",
    )
    severity: str = Field(
        default="warning",
        description="Current incident severity: info, warning, critical, or resolved",
    )


class IncidentResponseState(State):
    """Episode state for the incident response environment.

    Inherits episode_id (Optional[str]) and step_count (int) from State.
    """

    task_id: str = Field(default="", description="Which scenario is loaded")
    difficulty: str = Field(default="easy", description="easy, medium, or hard")
    max_steps: int = Field(default=20, description="Maximum steps before timeout")
