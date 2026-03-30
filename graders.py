"""Deterministic graders for incident response scenarios.

Each grader produces a score between 0.0 and 1.0 based on:
- Investigation quality (30%): Did the agent check the right services?
- Diagnosis accuracy (25%): Did the agent identify the root cause via keyword match?
- Remediation correctness (30%): Did the agent take the correct fix(es)?
- Efficiency (15%): How many steps vs optimal?
"""

from typing import List, Set, Tuple

try:
    from .scenarios import Scenario
except ImportError:
    from scenarios import Scenario


def score_investigation(
    checks_performed: Set[Tuple[str, str]],
    scenario: Scenario,
) -> float:
    """Score how well the agent investigated.

    Args:
        checks_performed: Set of (command, service) tuples the agent executed
        scenario: The scenario being graded

    Returns:
        Score between 0.0 and 1.0
    """
    critical = set(scenario.critical_checks)
    if not critical:
        return 1.0

    hits = len(checks_performed & critical)
    return hits / len(critical)


def score_diagnosis(
    diagnosis_text: str,
    scenario: Scenario,
) -> float:
    """Score the agent's diagnosis based on keyword matching.

    Case-insensitive substring matching against required keywords.

    Args:
        diagnosis_text: The explanation text from the agent's diagnose command
        scenario: The scenario being graded

    Returns:
        Score between 0.0 and 1.0
    """
    if not scenario.diagnosis_keywords:
        return 1.0
    if not diagnosis_text:
        return 0.0

    text_lower = diagnosis_text.lower()
    matched = sum(1 for kw in scenario.diagnosis_keywords if kw.lower() in text_lower)
    return matched / len(scenario.diagnosis_keywords)


def score_remediation(
    remediations_taken: List[Tuple[str, str]],
    scenario: Scenario,
) -> float:
    """Score the agent's remediation actions.

    Awards credit for correct remediations, penalizes wrong ones.

    Args:
        remediations_taken: List of (command, target) tuples the agent executed
        scenario: The scenario being graded

    Returns:
        Score between 0.0 and 1.0
    """
    required = set(scenario.correct_remediations)
    if not required:
        return 1.0

    taken = set(remediations_taken)
    correct_count = len(taken & required)
    wrong_count = len(taken - required)

    base_score = correct_count / len(required)
    penalty = wrong_count * 0.15
    return max(0.0, min(1.0, base_score - penalty))


def score_efficiency(
    actual_steps: int,
    scenario: Scenario,
) -> float:
    """Score how efficiently the agent solved the problem.

    Perfect score if agent uses optimal number of steps.
    Score decreases linearly toward 0 as steps approach max.

    Args:
        actual_steps: Number of steps the agent took
        scenario: The scenario being graded

    Returns:
        Score between 0.0 and 1.0
    """
    if actual_steps <= scenario.optimal_steps:
        return 1.0

    denominator = scenario.max_steps - scenario.optimal_steps
    if denominator <= 0:
        return 1.0

    return max(0.0, 1.0 - (actual_steps - scenario.optimal_steps) / denominator)


def compute_final_score(
    checks_performed: Set[Tuple[str, str]],
    diagnosis_text: str,
    remediations_taken: List[Tuple[str, str]],
    actual_steps: int,
    scenario: Scenario,
) -> float:
    """Compute the final composite score (0.0 to 1.0).

    Weighted:
        30% investigation + 25% diagnosis + 30% remediation + 15% efficiency

    Args:
        checks_performed: Set of (command, service) investigation tuples
        diagnosis_text: Text from agent's diagnose command
        remediations_taken: List of (command, target) remediation tuples
        actual_steps: Total steps taken
        scenario: The scenario being graded

    Returns:
        Final score between 0.0 and 1.0
    """
    inv = score_investigation(checks_performed, scenario)
    diag = score_diagnosis(diagnosis_text, scenario)
    rem = score_remediation(remediations_taken, scenario)
    eff = score_efficiency(actual_steps, scenario)

    return round(0.30 * inv + 0.25 * diag + 0.30 * rem + 0.15 * eff, 4)
