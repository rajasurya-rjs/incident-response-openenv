"""Incident Response Environment — core game logic.

Implements reset(), step(), and state for the on-call triage simulator.
"""

import random
from typing import Any, List, Optional, Set, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        IncidentResponseAction,
        IncidentResponseObservation,
        IncidentResponseState,
    )
    from ..infrastructure import (
        ALL_COMMANDS,
        INVESTIGATION_COMMANDS,
        REMEDIATION_COMMANDS,
        SERVICES,
        check_service_status,
        execute_remediation,
        generate_logs,
        generate_metrics,
    )
    from ..scenarios import SCENARIOS, TASK_IDS, Scenario
    from ..graders import compute_final_score
except ImportError:
    from models import (
        IncidentResponseAction,
        IncidentResponseObservation,
        IncidentResponseState,
    )
    from infrastructure import (
        ALL_COMMANDS,
        INVESTIGATION_COMMANDS,
        REMEDIATION_COMMANDS,
        SERVICES,
        check_service_status,
        execute_remediation,
        generate_logs,
        generate_metrics,
    )
    from scenarios import SCENARIOS, TASK_IDS, Scenario
    from graders import compute_final_score


class IncidentResponseEnvironment(Environment):
    """On-call incident response simulator.

    The agent receives alerts about infrastructure issues and must:
    1. INVESTIGATE — check metrics, read logs, check status of services
    2. DIAGNOSE — identify the root cause
    3. REMEDIATE — take the correct action to fix the issue

    Supports 4 scenarios ranging from easy to hard.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = IncidentResponseState(episode_id=str(uuid4()), step_count=0)
        self._scenario: Optional[Scenario] = None
        self._checks_performed: Set[Tuple[str, str]] = set()
        self._remediations_taken: List[Tuple[str, str]] = []
        self._actions_taken: List[str] = []
        self._services_investigated: List[str] = []
        self._diagnosis_text: str = ""
        self._done: bool = False
        self._cumulative_reward: float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> IncidentResponseObservation:
        """Reset the environment and start a new incident scenario.

        Args:
            seed: Random seed for reproducibility
            episode_id: Custom episode identifier
            **kwargs: Must include 'task_id' to select scenario.
                      If not provided, a random scenario is selected.

        Returns:
            Initial observation with the alert summary.
        """
        if seed is not None:
            random.seed(seed)

        task_id = kwargs.get("task_id", None)
        if task_id is None:
            task_id = random.choice(TASK_IDS)

        if task_id not in SCENARIOS:
            task_id = TASK_IDS[0]

        self._scenario = SCENARIOS[task_id]
        self._checks_performed = set()
        self._remediations_taken = []
        self._actions_taken = []
        self._services_investigated = []
        self._diagnosis_text = ""
        self._done = False
        self._cumulative_reward = 0.0

        self._state = IncidentResponseState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            difficulty=self._scenario.difficulty,
            max_steps=self._scenario.max_steps,
        )

        return IncidentResponseObservation(
            done=False,
            reward=0.01,
            alert_summary=self._scenario.alert_text,
            command_output="Incident channel opened. You are the on-call engineer. Investigate and resolve this incident.",
            available_commands=list(ALL_COMMANDS),
            available_services=list(SERVICES),
            services_investigated=[],
            actions_taken=[],
            time_elapsed_minutes=0,
            severity=self._scenario.alert_severity,
        )

    def step(
        self,
        action: IncidentResponseAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> IncidentResponseObservation:
        """Execute one step in the incident response.

        Args:
            action: The agent's action (command + target + parameters)
            timeout_s: Optional timeout (unused)

        Returns:
            Observation with the result of the action.
        """
        if self._done:
            return self._make_terminal_observation(
                "Episode already complete. Call reset() to start a new incident."
            )

        if self._scenario is None:
            return self._make_terminal_observation(
                "No scenario loaded. Call reset() first."
            )

        self._state.step_count += 1
        command = action.command.strip().lower()
        target = action.target.strip().lower()

        # Track action in history
        action_str = f"{command} {target}".strip()
        self._actions_taken.append(action_str)

        # Validate command
        if command not in ALL_COMMANDS:
            return self._make_observation(
                f"Unknown command: '{command}'. Available commands: {', '.join(ALL_COMMANDS)}",
                reward=-0.01,
            )

        # Dispatch
        if command in INVESTIGATION_COMMANDS:
            return self._handle_investigation(command, target)
        elif command in REMEDIATION_COMMANDS:
            return self._handle_remediation(command, target)
        elif command == "diagnose":
            return self._handle_diagnosis(action.parameters)
        elif command == "escalate":
            return self._handle_escalation()

        return self._make_observation(
            f"Command '{command}' not recognized.",
            reward=-0.01,
        )

    @property
    def state(self) -> IncidentResponseState:
        """Get current episode state."""
        return self._state

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    def _handle_investigation(self, command: str, target: str) -> IncidentResponseObservation:
        """Handle check_metrics, read_logs, check_status commands."""
        if not target:
            return self._make_observation(
                f"'{command}' requires a target service. Available: {', '.join(SERVICES)}",
                reward=-0.005,
            )

        if target not in SERVICES:
            return self._make_observation(
                f"Unknown service: '{target}'. Available: {', '.join(SERVICES)}",
                reward=-0.005,
            )

        scenario_id = self._scenario.id
        check_key = (command, target)

        # Generate output based on command type
        if command == "check_metrics":
            output = generate_metrics(target, scenario_id)
        elif command == "read_logs":
            output = generate_logs(target, scenario_id)
        elif command == "check_status":
            output = check_service_status(target, scenario_id)
        else:
            output = f"Unknown investigation command: {command}"

        # Track investigation
        if target not in self._services_investigated:
            self._services_investigated.append(target)

        # Calculate reward
        already_checked = check_key in self._checks_performed
        self._checks_performed.add(check_key)

        if already_checked:
            reward = 0.0  # No reward for re-checking
        elif check_key in set(self._scenario.critical_checks):
            reward = 0.10  # Critical check
        elif target in self._scenario.relevant_services:
            reward = 0.05  # Relevant service
        else:
            reward = 0.01  # Exploring

        reward -= 0.005  # Time cost

        # Check if max steps reached
        if self._state.step_count >= self._scenario.max_steps:
            return self._make_terminal_with_score(
                output + "\n\nMax steps reached. Episode ending.",
            )

        return self._make_observation(output, reward=reward)

    def _handle_remediation(self, command: str, target: str) -> IncidentResponseObservation:
        """Handle restart_service, rollback_deploy, scale_service, clear_disk commands."""
        if not target:
            return self._make_observation(
                f"'{command}' requires a target service. Available: {', '.join(SERVICES)}",
                reward=-0.005,
            )

        output, is_correct = execute_remediation(
            command, target, self._scenario.id, self._scenario.correct_remediations
        )

        self._remediations_taken.append((command, target))

        if is_correct:
            reward = 0.25
        else:
            reward = -0.10

        reward -= 0.005  # Time cost

        # Check if all required remediations are done
        required = set(self._scenario.correct_remediations)
        taken = set(self._remediations_taken)
        all_done = required.issubset(taken)

        if all_done:
            return self._make_terminal_with_score(
                output + "\n\nAll required remediations completed. Incident resolved!",
            )

        # Check max steps
        if self._state.step_count >= self._scenario.max_steps:
            return self._make_terminal_with_score(
                output + "\n\nMax steps reached. Episode ending.",
            )

        return self._make_observation(output, reward=reward)

    def _handle_diagnosis(self, parameters: dict) -> IncidentResponseObservation:
        """Handle the diagnose command — agent submits root cause analysis."""
        explanation = parameters.get("explanation", "")
        if not explanation:
            return self._make_observation(
                "The 'diagnose' command requires parameters={'explanation': 'your root cause analysis'}.",
                reward=-0.005,
            )

        self._diagnosis_text = explanation

        # Score diagnosis via keyword matching
        matched = sum(
            1 for kw in self._scenario.diagnosis_keywords
            if kw.lower() in explanation.lower()
        )
        total = len(self._scenario.diagnosis_keywords)
        diag_score = matched / total if total > 0 else 0.0

        reward = 0.20 * diag_score - 0.005  # Partial credit for diagnosis

        output = (
            f"Diagnosis submitted.\n"
            f"Root cause analysis recorded: \"{explanation[:200]}{'...' if len(explanation) > 200 else ''}\"\n"
            f"Continue with remediation actions to resolve the incident."
        )

        # Check max steps
        if self._state.step_count >= self._scenario.max_steps:
            return self._make_terminal_with_score(
                output + "\n\nMax steps reached. Episode ending.",
            )

        return self._make_observation(output, reward=reward)

    def _handle_escalation(self) -> IncidentResponseObservation:
        """Handle escalation — agent gives up and pages senior engineer."""
        # Partial credit based on investigation done and difficulty
        inv_done = len(self._checks_performed & set(self._scenario.critical_checks))
        inv_total = len(self._scenario.critical_checks)
        inv_ratio = inv_done / inv_total if inv_total > 0 else 0.0

        if self._scenario.difficulty == "hard":
            base_escalation_score = 0.30
        elif self._scenario.difficulty == "medium":
            base_escalation_score = 0.20
        else:
            base_escalation_score = 0.10

        # Reward investigation done before escalating
        escalation_reward = base_escalation_score * inv_ratio

        output = (
            f"Escalated to senior on-call engineer.\n"
            f"Investigation summary: checked {inv_done}/{inv_total} critical services.\n"
            f"Escalation is a valid action when unsure of the root cause."
        )

        return self._make_terminal_with_score(output)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_observation(
        self, command_output: str, reward: float = 0.0
    ) -> IncidentResponseObservation:
        """Create a non-terminal observation."""
        safe_r = max(0.01, min(0.99, reward)) if reward is not None else 0.01
        self._cumulative_reward += safe_r
        time_elapsed = self._state.step_count * 2  # ~2 min per step

        return IncidentResponseObservation(
            done=False,
            reward=round(safe_r, 4),
            alert_summary=self._scenario.alert_text,
            command_output=command_output,
            available_commands=list(ALL_COMMANDS),
            available_services=list(SERVICES),
            services_investigated=list(self._services_investigated),
            actions_taken=list(self._actions_taken),
            time_elapsed_minutes=time_elapsed,
            severity=self._scenario.alert_severity,
        )

    def _make_terminal_observation(self, message: str) -> IncidentResponseObservation:
        """Create a terminal observation (episode over, no score computation)."""
        return IncidentResponseObservation(
            done=True,
            reward=0.01,  # Never 0.0 — evaluator requires strictly open interval (0,1)
            alert_summary=self._scenario.alert_text if self._scenario else "",
            command_output=message,
            available_commands=[],
            available_services=[],
            services_investigated=list(self._services_investigated),
            actions_taken=list(self._actions_taken),
            time_elapsed_minutes=self._state.step_count * 2,
            severity="resolved",
        )

    def _make_terminal_with_score(self, message: str) -> IncidentResponseObservation:
        """Create a terminal observation with final graded score."""
        self._done = True

        final_score = compute_final_score(
            checks_performed=self._checks_performed,
            diagnosis_text=self._diagnosis_text,
            remediations_taken=self._remediations_taken,
            actual_steps=self._state.step_count,
            scenario=self._scenario,
        )

        full_message = (
            f"{message}\n\n"
            f"=== Episode Complete ===\n"
            f"Task: {self._scenario.id} ({self._scenario.difficulty})\n"
            f"Steps taken: {self._state.step_count}/{self._scenario.max_steps}\n"
            f"Final score: {final_score:.4f}"
        )

        return IncidentResponseObservation(
            done=True,
            reward=final_score,
            alert_summary=self._scenario.alert_text,
            command_output=full_message,
            available_commands=[],
            available_services=[],
            services_investigated=list(self._services_investigated),
            actions_taken=list(self._actions_taken),
            time_elapsed_minutes=self._state.step_count * 2,
            severity="resolved",
        )
