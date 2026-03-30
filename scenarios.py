"""Scenario definitions for the incident response environment.

Each scenario defines a concrete incident with:
- An initial alert
- A root cause
- Required investigation checks
- Diagnosis keywords (for deterministic keyword-match grading)
- Correct remediation actions
- Difficulty and step limits
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Scenario:
    """A single incident scenario."""

    id: str
    difficulty: str  # "easy", "medium", "hard"
    alert_text: str
    alert_severity: str  # "info", "warning", "critical"

    # Root cause (hidden from agent, used for grading context)
    root_cause: str

    # Investigation checks that are critical for diagnosis
    # List of (command, service) tuples the agent should perform
    critical_checks: List[Tuple[str, str]]

    # Services that are relevant to this scenario (investigating these gives partial credit)
    relevant_services: List[str]

    # Keywords the agent should mention in their diagnosis
    # Grader does case-insensitive substring matching
    diagnosis_keywords: List[str]

    # Correct remediation actions: list of (command, target) tuples
    correct_remediations: List[Tuple[str, str]]

    # Step limits
    optimal_steps: int
    max_steps: int


# ---------------------------------------------------------------------------
# Task 1: Disk Full on Database — EASY
# ---------------------------------------------------------------------------
DISK_FULL = Scenario(
    id="disk_full",
    difficulty="easy",
    alert_text=(
        "CRITICAL ALERT: Disk usage on postgres-primary has exceeded 95% threshold.\n"
        "Current disk usage: 97%. Service: postgres-primary.\n"
        "Impact: Database writes may fail. WAL archiving has stopped.\n"
        "Time: 2024-01-15 14:35:00 UTC"
    ),
    alert_severity="critical",
    root_cause="WAL segments not being archived and temp tables bloated, filling up postgres-primary disk to 97%",
    critical_checks=[
        ("check_metrics", "postgres-primary"),
        ("read_logs", "postgres-primary"),
    ],
    relevant_services=["postgres-primary", "user-service"],
    diagnosis_keywords=["disk", "postgres", "wal", "space"],
    correct_remediations=[("clear_disk", "postgres-primary")],
    optimal_steps=3,
    max_steps=15,
)

# ---------------------------------------------------------------------------
# Task 2: Cascading Failure from Bad Deploy — MEDIUM
# ---------------------------------------------------------------------------
BAD_DEPLOY = Scenario(
    id="bad_deploy",
    difficulty="medium",
    alert_text=(
        "WARNING ALERT: Error rate on api-gateway has exceeded 5% threshold.\n"
        "Current error rate: 8.2%. Affected endpoints: /api/v1/checkout.\n"
        "Impact: Checkout requests failing with 502 Bad Gateway.\n"
        "Time: 2024-01-15 14:35:00 UTC"
    ),
    alert_severity="warning",
    root_cause=(
        "Bad deployment v2.3.1 on order-service introduced a NullPointerException "
        "in CheckoutHandler. This causes 500 errors on /checkout, which api-gateway "
        "reports as 502 Bad Gateway. Retry storms are overloading postgres-primary "
        "connection pool."
    ),
    critical_checks=[
        ("check_metrics", "api-gateway"),
        ("read_logs", "api-gateway"),
        ("check_metrics", "order-service"),
        ("read_logs", "order-service"),
    ],
    relevant_services=["api-gateway", "order-service", "postgres-primary"],
    diagnosis_keywords=["deploy", "order-service", "checkout", "nullpointer"],
    correct_remediations=[("rollback_deploy", "order-service")],
    optimal_steps=5,
    max_steps=20,
)

# ---------------------------------------------------------------------------
# Task 3: Memory Leak + Stale Cache — HARD
# ---------------------------------------------------------------------------
MEMORY_AND_CACHE = Scenario(
    id="memory_and_cache",
    difficulty="hard",
    alert_text=(
        "WARNING ALERT: p99 latency on api-gateway has exceeded 2000ms threshold.\n"
        "Current p99 latency: 2300ms. Error rate: 1.2% (within threshold).\n"
        "Impact: User-facing requests are slow. Intermittent timeouts on /users/* endpoints.\n"
        "Time: 2024-01-15 14:35:00 UTC"
    ),
    alert_severity="warning",
    root_cause=(
        "Two concurrent issues: (1) user-service has a memory leak causing repeated "
        "OOM kills and GC pauses of 900-1200ms, and (2) redis-cache has TTL disabled "
        "for user:profile:* keys since Jan 12, causing all profile lookups to return "
        "stale data (72+ hours old). The high hit rate (99.5%) is misleading — it "
        "looks healthy but every hit returns stale data."
    ),
    critical_checks=[
        ("check_metrics", "user-service"),
        ("read_logs", "user-service"),
        ("check_metrics", "redis-cache"),
        ("read_logs", "redis-cache"),
        ("check_metrics", "api-gateway"),
    ],
    relevant_services=["api-gateway", "user-service", "redis-cache", "order-service"],
    diagnosis_keywords=["memory", "leak", "user-service", "redis", "stale", "ttl", "cache"],
    correct_remediations=[
        ("restart_service", "user-service"),
        ("restart_service", "redis-cache"),
    ],
    optimal_steps=8,
    max_steps=25,
)

# ---------------------------------------------------------------------------
# Task 4: Kafka Consumer Lag Storm — MEDIUM-HARD
# ---------------------------------------------------------------------------
KAFKA_LAG = Scenario(
    id="kafka_lag",
    difficulty="medium",
    alert_text=(
        "CRITICAL ALERT: Consumer lag on kafka-broker has exceeded 100,000 messages.\n"
        "Current consumer lag: 104,382 messages across 4 partitions.\n"
        "Consumer group: order-consumers. Active consumers: 1 (expected: 4).\n"
        "Impact: Order processing delayed by ~12 minutes and growing.\n"
        "Time: 2024-01-15 14:35:00 UTC"
    ),
    alert_severity="critical",
    root_cause=(
        "During a maintenance window (2024-01-15 02:00-04:00 UTC), order-service "
        "was scaled from 4 replicas to 1. The scale-back was never executed after "
        "maintenance completed. Meanwhile, a marketing campaign drove 3.3x normal "
        "traffic (2800/s vs normal 850/s). With only 1 consumer, the lag is growing."
    ),
    critical_checks=[
        ("check_metrics", "kafka-broker"),
        ("read_logs", "kafka-broker"),
        ("check_metrics", "order-service"),
        ("read_logs", "order-service"),
    ],
    relevant_services=["kafka-broker", "order-service", "api-gateway"],
    diagnosis_keywords=["scale", "consumer", "order-service", "maintenance", "replica"],
    correct_remediations=[("scale_service", "order-service")],
    optimal_steps=5,
    max_steps=20,
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, Scenario] = {
    "disk_full": DISK_FULL,
    "bad_deploy": BAD_DEPLOY,
    "memory_and_cache": MEMORY_AND_CACHE,
    "kafka_lag": KAFKA_LAG,
}

TASK_IDS = list(SCENARIOS.keys())
