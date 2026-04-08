"""Deterministic graders for incident response scenarios.

Each grader produces a score strictly between 0.0 and 1.0 based on:
- Investigation quality (30%): Did the agent check the right services?
- Diagnosis accuracy (25%): Did the agent identify the root cause via keyword match?
- Remediation correctness (30%): Did the agent take the correct fix(es)?
- Efficiency (15%): How many steps vs optimal?

All functions clamp to (0.01, 0.99) — evaluator requires strictly open interval (0, 1).
"""

from typing import List, Set, Tuple

try:
    from .scenarios import Scenario
except ImportError:
    from scenarios import Scenario

# Bounds that format safely as "0.01" / "0.99" with :.2f
_MIN = 0.01
_MAX = 0.99


def score_investigation(
    checks_performed: Set[Tuple[str, str]],
    scenario: Scenario,
) -> float:
    """Score how well the agent investigated. Returns value in (_MIN, _MAX)."""
    critical = set(scenario.critical_checks)
    if not critical:
        return _MAX

    hits = len(checks_performed & critical)
    raw = hits / len(critical)
    return max(_MIN, min(_MAX, raw))


def score_diagnosis(
    diagnosis_text: str,
    scenario: Scenario,
) -> float:
    """Score the agent's diagnosis based on keyword matching. Returns value in (_MIN, _MAX)."""
    if not scenario.diagnosis_keywords:
        return _MAX

    if not diagnosis_text:
        return _MIN

    text_lower = diagnosis_text.lower()
    matched = sum(1 for kw in scenario.diagnosis_keywords if kw.lower() in text_lower)
    raw = matched / len(scenario.diagnosis_keywords)
    return max(_MIN, min(_MAX, raw))


def score_remediation(
    remediations_taken: List[Tuple[str, str]],
    scenario: Scenario,
) -> float:
    """Score the agent's remediation actions. Returns value in (_MIN, _MAX)."""
    required = set(scenario.correct_remediations)
    if not required:
        return _MAX

    taken = set(remediations_taken)
    correct_count = len(taken & required)
    wrong_count = len(taken - required)

    base_score = correct_count / len(required)
    penalty = wrong_count * 0.15
    raw = base_score - penalty
    return max(_MIN, min(_MAX, raw))


def score_efficiency(
    actual_steps: int,
    scenario: Scenario,
) -> float:
    """Score how efficiently the agent solved the problem. Returns value in (_MIN, _MAX)."""
    if actual_steps <= scenario.optimal_steps:
        return _MAX  # Perfect efficiency → 0.99, not 1.0

    denominator = scenario.max_steps - scenario.optimal_steps
    if denominator <= 0:
        return _MAX

    raw = 1.0 - (actual_steps - scenario.optimal_steps) / denominator
    return max(_MIN, min(_MAX, raw))


def compute_final_score(
    checks_performed: Set[Tuple[str, str]],
    diagnosis_text: str,
    remediations_taken: List[Tuple[str, str]],
    actual_steps: int,
    scenario: Scenario,
) -> float:
    """Compute the final composite score strictly in (0.01, 0.99).

    Weighted:
        30% investigation + 25% diagnosis + 30% remediation + 15% efficiency
    """
    inv = score_investigation(checks_performed, scenario)
    diag = score_diagnosis(diagnosis_text, scenario)
    rem = score_remediation(remediations_taken, scenario)
    eff = score_efficiency(actual_steps, scenario)

    raw = 0.30 * inv + 0.25 * diag + 0.30 * rem + 0.15 * eff
    return round(max(_MIN, min(_MAX, raw)), 4)
