"""Simulated microservice infrastructure for incident response scenarios.

Generates deterministic metrics, logs, and service status for 6 services.
All outputs are deterministic given (scenario_id, service_name, step_count).
"""

from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Service definitions
# ---------------------------------------------------------------------------

SERVICES = [
    "api-gateway",
    "user-service",
    "order-service",
    "postgres-primary",
    "redis-cache",
    "kafka-broker",
]

INVESTIGATION_COMMANDS = ["check_metrics", "read_logs", "check_status"]
REMEDIATION_COMMANDS = [
    "restart_service",
    "scale_service",
    "rollback_deploy",
    "clear_disk",
]
ALL_COMMANDS = INVESTIGATION_COMMANDS + REMEDIATION_COMMANDS + ["diagnose", "escalate"]

# ---------------------------------------------------------------------------
# Baseline (healthy) metrics for each service
# ---------------------------------------------------------------------------

HEALTHY_METRICS: Dict[str, Dict[str, Any]] = {
    "api-gateway": {
        "request_rate": "1200 req/s",
        "error_rate": "0.3%",
        "p99_latency_ms": "180ms",
        "active_connections": 342,
    },
    "user-service": {
        "cpu_percent": "22%",
        "memory_percent": "45%",
        "request_latency_ms": "35ms",
        "error_count": "2/min",
    },
    "order-service": {
        "cpu_percent": "30%",
        "memory_percent": "40%",
        "queue_depth": 12,
        "processing_time_ms": "85ms",
        "failed_transactions": "0/min",
    },
    "postgres-primary": {
        "cpu_percent": "18%",
        "connections_active": 45,
        "connections_max": 200,
        "disk_usage_percent": "52%",
        "replication_lag_s": "0.2s",
    },
    "redis-cache": {
        "memory_used_mb": "512 MB",
        "memory_max_mb": "2048 MB",
        "hit_rate_percent": "94.2%",
        "eviction_rate": "3/min",
        "connected_clients": 28,
    },
    "kafka-broker": {
        "consumer_lag": 120,
        "partition_count": 24,
        "disk_usage_percent": "35%",
        "isr_count": 24,
        "messages_in_per_sec": "850/s",
    },
}

HEALTHY_STATUS: Dict[str, Dict[str, Any]] = {
    "api-gateway": {
        "status": "healthy",
        "uptime": "14d 7h 23m",
        "version": "v3.1.2",
        "last_deploy": "2024-01-10 09:15:00 UTC",
        "restarts_24h": 0,
    },
    "user-service": {
        "status": "healthy",
        "uptime": "14d 7h 23m",
        "version": "v2.8.0",
        "last_deploy": "2024-01-08 14:30:00 UTC",
        "restarts_24h": 0,
    },
    "order-service": {
        "status": "healthy",
        "uptime": "14d 7h 23m",
        "version": "v2.3.0",
        "last_deploy": "2024-01-09 11:00:00 UTC",
        "restarts_24h": 0,
    },
    "postgres-primary": {
        "status": "healthy",
        "uptime": "45d 2h 10m",
        "version": "PostgreSQL 15.4",
        "last_deploy": "N/A",
        "restarts_24h": 0,
    },
    "redis-cache": {
        "status": "healthy",
        "uptime": "30d 5h 45m",
        "version": "Redis 7.2.3",
        "last_deploy": "N/A",
        "restarts_24h": 0,
    },
    "kafka-broker": {
        "status": "healthy",
        "uptime": "60d 12h 5m",
        "version": "Kafka 3.6.1",
        "last_deploy": "N/A",
        "restarts_24h": 0,
    },
}

HEALTHY_LOGS: Dict[str, List[str]] = {
    "api-gateway": [
        "[14:30:01] 200 OK → GET /api/v1/users (12ms)",
        "[14:30:01] 200 OK → GET /api/v1/products (18ms)",
        "[14:30:02] 200 OK → POST /api/v1/orders (45ms)",
        "[14:30:02] 200 OK → GET /api/v1/health (2ms)",
        "[14:30:03] 200 OK → GET /api/v1/users/profile (15ms)",
    ],
    "user-service": [
        "[14:30:01] INFO  AuthHandler: Token validated for user_id=8821",
        "[14:30:01] INFO  ProfileService: Cache hit for user profile user_id=8821",
        "[14:30:02] INFO  AuthHandler: Token validated for user_id=3345",
        "[14:30:02] INFO  SessionManager: Session refreshed sid=a8f2e1",
        "[14:30:03] INFO  AuthHandler: Token validated for user_id=1120",
    ],
    "order-service": [
        "[14:30:01] INFO  OrderProcessor: Order #98201 processed successfully (82ms)",
        "[14:30:02] INFO  PaymentGateway: Payment confirmed for order #98201",
        "[14:30:02] INFO  InventoryService: Stock updated for SKU-4421",
        "[14:30:03] INFO  OrderProcessor: Order #98202 processing started",
        "[14:30:03] INFO  OrderProcessor: Order #98202 processed successfully (91ms)",
    ],
    "postgres-primary": [
        "[14:30:01] LOG:  duration: 2.3ms  statement: SELECT * FROM users WHERE id = 8821",
        "[14:30:02] LOG:  duration: 1.1ms  statement: SELECT * FROM orders WHERE user_id = 8821",
        "[14:30:02] LOG:  duration: 5.8ms  statement: INSERT INTO orders VALUES (...)",
        "[14:30:03] LOG:  checkpoint starting: time",
        "[14:30:03] LOG:  checkpoint complete: wrote 42 buffers (0.3%)",
    ],
    "redis-cache": [
        "[14:30:01] INFO  GET user:profile:8821 - HIT (0.2ms)",
        "[14:30:01] INFO  GET session:a8f2e1 - HIT (0.1ms)",
        "[14:30:02] INFO  SET order:cache:98201 - OK (0.3ms)",
        "[14:30:02] INFO  EXPIRE order:cache:98201 3600 - OK",
        "[14:30:03] INFO  GET user:profile:1120 - HIT (0.2ms)",
    ],
    "kafka-broker": [
        "[14:30:01] INFO  [GroupCoordinator] Group order-consumers: stable (3 members)",
        "[14:30:02] INFO  [Partition orders-0] High watermark: 1284021",
        "[14:30:02] INFO  [Partition orders-1] High watermark: 1283998",
        "[14:30:03] INFO  [ReplicaManager] All partitions in-sync",
        "[14:30:03] INFO  [LogCleaner] Cleaned segment orders-0: 12MB freed",
    ],
}


# ---------------------------------------------------------------------------
# Scenario-specific overrides
# ---------------------------------------------------------------------------


def _format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metrics dict as a readable string."""
    lines = []
    for key, value in metrics.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _format_status(status_info: Dict[str, Any]) -> str:
    """Format status dict as a readable string."""
    lines = []
    for key, value in status_info.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _format_logs(log_lines: List[str]) -> str:
    """Format log lines as a readable block."""
    return "\n".join(log_lines)


# ---------------------------------------------------------------------------
# Scenario metric overrides — what changes from healthy baselines
# ---------------------------------------------------------------------------

SCENARIO_METRIC_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "disk_full": {
        "postgres-primary": {
            "cpu_percent": "35%",
            "connections_active": 48,
            "disk_usage_percent": "97%",
            "replication_lag_s": "12.5s",
        },
        "user-service": {
            "request_latency_ms": "120ms",
        },
    },
    "bad_deploy": {
        "api-gateway": {
            "error_rate": "8.2%",
            "p99_latency_ms": "1800ms",
        },
        "order-service": {
            "cpu_percent": "92%",
            "queue_depth": 8420,
            "processing_time_ms": "timeout",
            "failed_transactions": "340/min",
        },
        "postgres-primary": {
            "connections_active": 185,
            "connections_max": 200,
            "cpu_percent": "68%",
        },
        "redis-cache": {
            "hit_rate_percent": "78.1%",
        },
    },
    "memory_and_cache": {
        "api-gateway": {
            "p99_latency_ms": "2300ms",
            "error_rate": "1.2%",
        },
        "user-service": {
            "cpu_percent": "45%",
            "memory_percent": "87%",
            "request_latency_ms": "850ms",
        },
        "redis-cache": {
            "hit_rate_percent": "99.5%",
            "eviction_rate": "0/min",
        },
        "kafka-broker": {
            "consumer_lag": 2400,
        },
        "order-service": {
            "cpu_percent": "38%",
            "processing_time_ms": "140ms",
        },
    },
    "kafka_lag": {
        "kafka-broker": {
            "consumer_lag": 104382,
            "messages_in_per_sec": "2800/s",
        },
        "order-service": {
            "cpu_percent": "95%",
            "queue_depth": 48000,
            "processing_time_ms": "2400ms",
        },
        "api-gateway": {
            "p99_latency_ms": "650ms",
        },
    },
}

# ---------------------------------------------------------------------------
# Scenario log overrides
# ---------------------------------------------------------------------------

SCENARIO_LOG_OVERRIDES: Dict[str, Dict[str, List[str]]] = {
    "disk_full": {
        "postgres-primary": [
            "[14:35:01] ERROR: could not write to file 'pg_wal/000000010000004200000038': No space left on device",
            "[14:35:01] LOG:   WAL segment 000000010000004200000038 cannot be archived: No space left on device",
            "[14:35:02] WARNING: disk usage on /var/lib/postgresql/data is at 97%",
            "[14:35:02] ERROR: could not extend file 'base/16384/pg_temp_3.12': No space left on device",
            "[14:35:03] LOG:   archiver process failed on 000000010000004200000035",
            "[14:35:03] WARNING: oldest WAL segment not yet archived: 000000010000004200000020 (3 days old)",
            "[14:35:04] ERROR: temporary file size exceeds temp_file_limit (2GB)",
            "[14:35:04] LOG:   automatic vacuum of table 'orders': could not extend — no space left",
        ],
        "user-service": [
            "[14:35:01] INFO  AuthHandler: Token validated for user_id=8821",
            "[14:35:01] WARN  ProfileService: Slow query detected (115ms) — database latency elevated",
            "[14:35:02] INFO  AuthHandler: Token validated for user_id=3345",
            "[14:35:02] INFO  ProfileService: Cache hit for user profile user_id=3345",
            "[14:35:03] WARN  ProfileService: Slow query detected (108ms) — database latency elevated",
        ],
    },
    "bad_deploy": {
        "api-gateway": [
            "[14:35:01] 502 Bad Gateway → POST /api/v1/checkout (upstream: order-service, timeout after 30s)",
            "[14:35:01] 200 OK → GET /api/v1/products (15ms)",
            "[14:35:01] 502 Bad Gateway → POST /api/v1/checkout (upstream: order-service, connection refused)",
            "[14:35:02] 502 Bad Gateway → POST /api/v1/checkout (upstream: order-service, timeout after 30s)",
            "[14:35:02] 200 OK → GET /api/v1/users (12ms)",
            "[14:35:02] WARN  UpstreamRetry: Retrying POST /checkout to order-service (attempt 3/3)",
            "[14:35:03] 502 Bad Gateway → POST /api/v1/checkout (upstream: order-service, connection refused)",
            "[14:35:03] WARN  CircuitBreaker: order-service /checkout error rate 42% — circuit OPEN",
        ],
        "order-service": [
            "[14:32:00] INFO  DeployManager: Deploy v2.3.1 started (commit: a4f8c2e)",
            "[14:32:15] INFO  DeployManager: Deploy v2.3.1 completed — health check passed",
            "[14:33:01] ERROR CheckoutHandler: NullPointerException in CheckoutHandler.process()",
            "[14:33:01] ERROR   at com.shop.checkout.CheckoutHandler.process(CheckoutHandler.java:142)",
            "[14:33:01] ERROR   at com.shop.checkout.CheckoutHandler.validateCart(CheckoutHandler.java:89)",
            "[14:33:02] ERROR CheckoutHandler: NullPointerException in CheckoutHandler.process()",
            "[14:33:02] ERROR CheckoutHandler: NullPointerException in CheckoutHandler.process()",
            "[14:33:03] WARN  HealthCheck: /checkout endpoint returning 500 — error rate 42%",
        ],
        "postgres-primary": [
            "[14:33:01] LOG:  connection received: host=order-service port=45821",
            "[14:33:01] LOG:  connection received: host=order-service port=45822",
            "[14:33:02] WARNING: too many connections: active=185, max=200",
            "[14:33:02] LOG:  connection received: host=api-gateway port=38291 (retry)",
            "[14:33:03] WARNING: too many connections: active=188, max=200",
            "[14:33:03] LOG:  duration: 45.2ms  statement: SELECT * FROM orders (slow due to connection contention)",
            "[14:33:04] FATAL: remaining connection slots reserved for superuser (3 left)",
            "[14:33:04] LOG:  connection pool exhaustion detected — consider increasing max_connections",
        ],
        "redis-cache": [
            "[14:33:01] INFO  GET order:cache:checkout_config - MISS",
            "[14:33:01] INFO  GET order:cache:checkout_config - MISS",
            "[14:33:02] INFO  GET session:b2c1d3 - HIT (0.2ms)",
            "[14:33:02] WARN  Hit rate dropped: 94.2% → 78.1% (checkout cache invalidated by errors)",
            "[14:33:03] INFO  GET user:profile:3345 - HIT (0.2ms)",
        ],
    },
    "memory_and_cache": {
        "api-gateway": [
            "[14:35:01] 200 OK → GET /api/v1/users/profile (1850ms) — SLOW",
            "[14:35:01] 200 OK → GET /api/v1/products (18ms)",
            "[14:35:02] 200 OK → POST /api/v1/orders (210ms)",
            "[14:35:02] 200 OK → GET /api/v1/users/profile (2100ms) — SLOW",
            "[14:35:03] 504 Gateway Timeout → GET /api/v1/users/profile (upstream: user-service, timeout 3s)",
            "[14:35:03] WARN  Latency: p99 for /users/* endpoints at 2300ms (threshold: 500ms)",
        ],
        "user-service": [
            "[14:30:00] WARN  JVM GC pause: 450ms (heap: 82% used)",
            "[14:31:15] WARN  JVM GC pause: 890ms (heap: 85% used)",
            "[14:32:30] ERROR OOMKilled: Container restarted (memory limit: 2048MB, used: 2048MB)",
            "[14:32:35] INFO  Service restarted — initializing connection pools",
            "[14:33:00] WARN  JVM GC pause: 1200ms (heap: 84% used)",
            "[14:34:00] WARN  Memory usage climbing: 1740MB / 2048MB (85%)",
            "[14:35:00] WARN  JVM GC pause: 950ms (heap: 87% used)",
            "[14:35:30] ERROR OOMKilled: Container restarted (memory limit: 2048MB, used: 2048MB)",
        ],
        "redis-cache": [
            "[14:30:01] INFO  GET user:profile:8821 - HIT (0.2ms) [age: 72h 15m]",
            "[14:30:01] INFO  GET user:profile:3345 - HIT (0.1ms) [age: 71h 42m]",
            "[14:30:02] WARN  TTL disabled for key pattern user:profile:* (config change 2024-01-12 02:00 UTC)",
            "[14:30:02] INFO  GET session:a8f2e1 - HIT (0.1ms) [age: 0h 5m]",
            "[14:30:03] INFO  Cache statistics: hit_rate=99.5%, total_keys=48201, keys_without_ttl=31204",
            "[14:30:03] WARN  31204 keys have no TTL set — potential stale data risk",
            "[14:30:04] INFO  GET user:profile:1120 - HIT (0.2ms) [age: 73h 01m]",
            "[14:30:04] WARN  Oldest cached entry: user:profile:5521 age=74h 12m (no expiry set)",
        ],
        "kafka-broker": [
            "[14:30:01] INFO  [GroupCoordinator] Group order-consumers: stable (3 members)",
            "[14:30:02] INFO  [Partition orders-0] Consumer lag: 2400 (elevated — consumers slow)",
            "[14:30:02] INFO  [Partition orders-1] High watermark: 1285102",
            "[14:30:03] INFO  [ReplicaManager] All partitions in-sync",
            "[14:30:03] INFO  [LogCleaner] Cleaned segment orders-0: 8MB freed",
        ],
        "order-service": [
            "[14:30:01] INFO  OrderProcessor: Order #98301 processed successfully (135ms)",
            "[14:30:02] WARN  ProfileLookup: Received stale user profile data for user_id=8821 (last updated 72h ago)",
            "[14:30:02] INFO  OrderProcessor: Order #98302 processed successfully (142ms)",
            "[14:30:03] WARN  ProfileLookup: Received stale user profile data for user_id=3345 (last updated 71h ago)",
            "[14:30:03] INFO  OrderProcessor: Order #98303 processed successfully (138ms)",
        ],
    },
    "kafka_lag": {
        "kafka-broker": [
            "[14:30:01] WARN  [GroupCoordinator] Group order-consumers: rebalancing (1 member — was 4)",
            "[14:30:01] WARN  [GroupCoordinator] Consumer group rebalance triggered: 3 members LEFT during maintenance window (2024-01-15 02:00-04:00 UTC)",
            "[14:30:02] WARN  [Partition orders-0] Consumer lag: 26,102 (CRITICAL — growing)",
            "[14:30:02] WARN  [Partition orders-1] Consumer lag: 25,884 (CRITICAL — growing)",
            "[14:30:03] WARN  [Partition orders-2] Consumer lag: 26,448 (CRITICAL — growing)",
            "[14:30:03] WARN  [Partition orders-3] Consumer lag: 25,948 (CRITICAL — growing)",
            "[14:30:04] INFO  [ProducerMetrics] Messages in: 2800/s (normal: 850/s — 3.3x traffic spike)",
            "[14:30:04] WARN  Consumer throughput: 280/s with 1 consumer (need ~4 consumers for current load)",
        ],
        "order-service": [
            "[14:30:01] WARN  ConsumerPool: Only 1 consumer instance active (expected: 4)",
            "[14:30:01] INFO  ScaleEvent: Scaled down from 4 → 1 replicas at 2024-01-15 02:00 UTC (maintenance)",
            "[14:30:02] WARN  QueueDepth: Processing queue at 48,000 items (threshold: 1000)",
            "[14:30:02] WARN  Processing backlog growing: estimated drain time at current rate: 47 hours",
            "[14:30:03] ERROR OrderProcessor: Order #95102 processing timeout (2400ms > 1000ms limit)",
            "[14:30:03] WARN  ConsumerPool: Consumer lag exceeding SLA — orders delayed by avg 12 minutes",
        ],
        "api-gateway": [
            "[14:30:01] 200 OK → POST /api/v1/orders (620ms) — SLOW",
            "[14:30:02] 200 OK → POST /api/v1/orders (680ms) — SLOW",
            "[14:30:02] 200 OK → GET /api/v1/products (15ms)",
            "[14:30:03] WARN  Upstream latency: order-service p99 at 650ms (threshold: 200ms)",
            "[14:30:03] 200 OK → GET /api/v1/users (12ms)",
        ],
    },
}

# ---------------------------------------------------------------------------
# Scenario status overrides
# ---------------------------------------------------------------------------

SCENARIO_STATUS_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "disk_full": {
        "postgres-primary": {
            "status": "degraded",
            "disk_partition": "/var/lib/postgresql/data: 97% used (185GB / 190GB)",
            "wal_archiving": "FAILED — no space left",
            "restarts_24h": 0,
        },
    },
    "bad_deploy": {
        "order-service": {
            "status": "degraded",
            "version": "v2.3.1",
            "last_deploy": "2024-01-15 14:32:00 UTC (23 minutes ago)",
            "previous_version": "v2.3.0",
            "health_check": "/checkout returning 500 (42% error rate)",
            "restarts_24h": 0,
        },
        "postgres-primary": {
            "status": "degraded",
            "connections": "185/200 active (92.5% — CRITICAL)",
            "restarts_24h": 0,
        },
    },
    "memory_and_cache": {
        "user-service": {
            "status": "unstable",
            "uptime": "0h 5m (restarted 3 times in last hour)",
            "restarts_24h": 3,
            "memory": "1740MB / 2048MB (85% — climbing)",
            "last_oom_kill": "2024-01-15 14:32:30 UTC (3 minutes ago)",
        },
        "redis-cache": {
            "status": "healthy",
            "uptime": "30d 5h 45m",
            "config_change": "TTL disabled for user:profile:* pattern at 2024-01-12 02:00 UTC",
            "keys_without_ttl": 31204,
            "restarts_24h": 0,
        },
    },
    "kafka_lag": {
        "kafka-broker": {
            "status": "degraded",
            "consumer_groups": "order-consumers: 1/4 members active",
            "lag_trend": "growing at ~500 messages/sec",
            "restarts_24h": 0,
        },
        "order-service": {
            "status": "degraded",
            "replicas": "1/4 active",
            "last_scale_event": "Scaled 4 → 1 at 2024-01-15 02:00 UTC (maintenance window)",
            "scale_back_scheduled": "No — manual action required",
            "restarts_24h": 0,
        },
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_metrics(service: str, scenario_id: str) -> str:
    """Generate deterministic metrics output for a service in a scenario."""
    if service not in SERVICES:
        return f"Error: Unknown service '{service}'. Available: {', '.join(SERVICES)}"

    base = dict(HEALTHY_METRICS[service])
    overrides = SCENARIO_METRIC_OVERRIDES.get(scenario_id, {}).get(service, {})
    base.update(overrides)

    header = f"=== Metrics: {service} ===\n"
    return header + _format_metrics(base)


def generate_logs(service: str, scenario_id: str) -> str:
    """Generate deterministic log output for a service in a scenario."""
    if service not in SERVICES:
        return f"Error: Unknown service '{service}'. Available: {', '.join(SERVICES)}"

    logs = SCENARIO_LOG_OVERRIDES.get(scenario_id, {}).get(service, HEALTHY_LOGS.get(service, []))

    header = f"=== Recent Logs: {service} ===\n"
    return header + _format_logs(logs)


def check_service_status(service: str, scenario_id: str) -> str:
    """Generate deterministic status output for a service in a scenario."""
    if service not in SERVICES:
        return f"Error: Unknown service '{service}'. Available: {', '.join(SERVICES)}"

    base = dict(HEALTHY_STATUS[service])
    overrides = SCENARIO_STATUS_OVERRIDES.get(scenario_id, {}).get(service, {})
    base.update(overrides)

    header = f"=== Status: {service} ===\n"
    return header + _format_status(base)


def execute_remediation(
    command: str, target: str, scenario_id: str, correct_remediations: List[Tuple[str, str]]
) -> Tuple[str, bool]:
    """Execute a remediation action. Returns (output_text, is_correct).

    Args:
        command: The remediation command (restart_service, rollback_deploy, etc.)
        target: Target service name
        scenario_id: Current scenario ID
        correct_remediations: List of (command, target) tuples that are correct for this scenario

    Returns:
        Tuple of (output message, whether this was a correct remediation)
    """
    if target not in SERVICES:
        return f"Error: Unknown service '{target}'. Available: {', '.join(SERVICES)}", False

    is_correct = (command, target) in correct_remediations

    if command == "restart_service":
        if is_correct:
            return (
                f"Restarting {target}...\n"
                f"  Stopping {target}: OK\n"
                f"  Starting {target}: OK\n"
                f"  Health check: PASSING\n"
                f"  {target} restarted successfully. Issue should be mitigated.",
                True,
            )
        else:
            return (
                f"Restarting {target}...\n"
                f"  Stopping {target}: OK\n"
                f"  Starting {target}: OK\n"
                f"  Health check: PASSING\n"
                f"  WARNING: {target} restarted but this service was not the root cause. "
                f"Unnecessary restart caused brief disruption to dependent services.",
                False,
            )

    elif command == "rollback_deploy":
        if is_correct:
            status = SCENARIO_STATUS_OVERRIDES.get(scenario_id, {}).get(target, {})
            prev = status.get("previous_version", "previous version")
            curr = status.get("version", "current version")
            return (
                f"Rolling back {target} from {curr} to {prev}...\n"
                f"  Deploying {prev}: OK\n"
                f"  Health check: PASSING\n"
                f"  Rollback successful. Error rate dropping.",
                True,
            )
        else:
            return (
                f"Rolling back {target}...\n"
                f"  No recent deployments found for {target} or rollback had no effect.\n"
                f"  WARNING: This service was not recently deployed. Rollback is a no-op.",
                False,
            )

    elif command == "scale_service":
        if is_correct:
            return (
                f"Scaling {target}...\n"
                f"  Current replicas: 1\n"
                f"  Scaling to: 4 replicas\n"
                f"  New instances starting: OK\n"
                f"  Consumer group rebalancing: OK\n"
                f"  {target} scaled successfully. Backlog should begin draining.",
                True,
            )
        else:
            return (
                f"Scaling {target}...\n"
                f"  Current replicas: 2\n"
                f"  Scaling to requested count: OK\n"
                f"  WARNING: {target} was not under-provisioned. Additional replicas are idle.",
                False,
            )

    elif command == "clear_disk":
        if is_correct:
            return (
                f"Clearing disk on {target}...\n"
                f"  Removing old WAL segments: 45GB freed\n"
                f"  Cleaning temp tables: 12GB freed\n"
                f"  Running VACUUM: OK\n"
                f"  Disk usage: 97% → 62%\n"
                f"  WAL archiving: RESUMED\n"
                f"  {target} disk cleaned successfully.",
                True,
            )
        else:
            return (
                f"Clearing disk on {target}...\n"
                f"  Disk usage: 52% (already healthy)\n"
                f"  WARNING: Disk was not full on {target}. No action needed.",
                False,
            )

    return f"Error: Unknown remediation command '{command}'.", False
