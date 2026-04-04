---
title: Incident Response Environment
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Incident Response / On-Call Triage Environment

An OpenEnv environment that simulates **real-world SRE incident response**. An AI agent receives infrastructure alerts across a 6-service microservice architecture, investigates by checking metrics and reading logs, diagnoses the root cause, and takes the correct remediation action.

## Why This Environment?

Every tech company has on-call engineers who get paged at 3 AM when something breaks. They must quickly:
1. **Investigate** — check metrics, read logs across multiple services
2. **Diagnose** — identify the root cause (which may be different from the alerting service)
3. **Remediate** — restart services, rollback deploys, scale up, or clean disk

This is a genuine unsolved problem — no existing RL environment trains agents for incident response. This environment fills that gap with realistic scenarios, cascading failures, red herrings, and deterministic grading.

## Action Space

```python
class IncidentResponseAction(Action):
    command: str     # What to do (see below)
    target: str      # Which service to act on
    parameters: dict # Extra params (e.g., explanation for diagnose)
```

### Available Commands

| Command | Type | Description |
|---------|------|-------------|
| `check_metrics` | Investigation | View CPU, memory, disk, error rates, latency for a service |
| `read_logs` | Investigation | Read recent application and system logs |
| `check_status` | Investigation | Check health status, uptime, deploy history |
| `restart_service` | Remediation | Restart a service |
| `scale_service` | Remediation | Scale up service replicas |
| `rollback_deploy` | Remediation | Roll back to previous deployment version |
| `clear_disk` | Remediation | Clean up disk space (WAL segments, temp files) |
| `diagnose` | Analysis | Submit root cause analysis (free-text explanation) |
| `escalate` | Escalation | Escalate to senior engineer (partial credit) |

### Available Services

| Service | Role |
|---------|------|
| `api-gateway` | Routes incoming HTTP traffic |
| `user-service` | Handles authentication and user profiles |
| `order-service` | Processes orders and payments |
| `postgres-primary` | Primary PostgreSQL database |
| `redis-cache` | Caching layer (sessions, profiles) |
| `kafka-broker` | Message queue for async processing |

## Observation Space

```python
class IncidentResponseObservation(Observation):
    alert_summary: str              # Current alert text
    command_output: str             # Result of the last command
    available_commands: List[str]   # Valid commands
    available_services: List[str]   # Valid service targets
    services_investigated: List[str]# Services already checked
    actions_taken: List[str]        # Action history
    time_elapsed_minutes: int       # Simulated time since alert
    severity: str                   # info/warning/critical/resolved
    # Inherited: done, reward
```

## Tasks

### Task 1: `disk_full` (Easy)
**Alert**: Disk usage on postgres-primary at 97%
**Root cause**: WAL segments not archived, temp tables bloated
**Expected**: Check metrics/logs on postgres, then clear disk. ~3 steps.

### Task 2: `bad_deploy` (Medium)
**Alert**: Error rate on api-gateway at 8%
**Root cause**: Bad deployment on order-service causes cascading 502 errors
**Expected**: Trace errors from api-gateway → order-service, then rollback. ~5 steps.
**Challenge**: postgres-primary looks sick (too many connections) but it's just a symptom.

### Task 3: `memory_and_cache` (Hard)
**Alert**: p99 latency on api-gateway at 2300ms
**Root cause**: TWO concurrent issues — user-service memory leak AND redis-cache serving stale data (TTL disabled, 99.5% hit rate is misleading)
**Expected**: Find both root causes, restart both services. ~8 steps.
**Challenge**: Redis appears healthy (high hit rate) but data is 72+ hours stale. Agent must reason about what "99.5% hit rate + no TTL" actually means.

### Task 4: `kafka_lag` (Medium-Hard)
**Alert**: Consumer lag on kafka-broker at 100k messages
**Root cause**: Consumers scaled down during maintenance, not restored. 3x traffic spike.
**Expected**: Identify scaling issue, scale up order-service. ~5 steps.

## Scoring

Each task is scored 0.0 to 1.0 using a deterministic weighted composite:

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| Investigation | 30% | Did the agent check the right services? |
| Diagnosis | 25% | Did the agent identify the root cause? (keyword matching) |
| Remediation | 30% | Did the agent take the correct fix? |
| Efficiency | 15% | Steps taken vs optimal (fewer = better) |

### Reward Shaping (Per-Step)
- Investigating a relevant service: **+0.05**
- Performing a critical check: **+0.10**
- Correct diagnosis keywords: **+0.20** (proportional)
- Correct remediation: **+0.25**
- Wrong remediation: **-0.10**
- Per-step time cost: **-0.005**

## Setup & Usage

### Quick Start (Remote)
```python
from incident_response_env import IncidentResponseEnv, IncidentResponseAction

with IncidentResponseEnv(base_url="https://<space-url>.hf.space").sync() as env:
    result = env.reset(task_id="disk_full")
    print(result.observation.alert_summary)

    result = env.step(IncidentResponseAction(
        command="check_metrics",
        target="postgres-primary",
    ))
    print(result.observation.command_output)
```

### Local Development
```bash
# Clone and install
git clone <repo-url>
cd incident_response_env
pip install -e .

# Run server
uvicorn incident_response_env.server.app:app --host 0.0.0.0 --port 8000

# Or with the entry point
python -m incident_response_env.server.app
```

### Docker
```bash
docker build -t incident-response-env .
docker run -d -p 8000:8000 incident-response-env

# Verify
curl http://localhost:8000/health
```

### Inference (Baseline)
```bash
# Start server first, then:
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
HF_TOKEN=hf_... \
python inference.py

# Required environment variables:
#   API_BASE_URL   The API endpoint for the LLM
#   MODEL_NAME     The model identifier to use for inference
#   HF_TOKEN       Your Hugging Face / API key
```

## Baseline Scores

Approximate scores (temperature=0):

| Task | Difficulty | Score |
|------|-----------|-------|
| `disk_full` | Easy | ~0.75 |
| `bad_deploy` | Medium | ~0.85 |
| `memory_and_cache` | Hard | ~0.60 |
| `kafka_lag` | Medium-Hard | ~0.80 |
| **Average** | | **~0.75** |

*Scores are approximate and may vary slightly between runs.*

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metadata` | GET | Environment metadata |
| `/schema` | GET | Action/observation/state schemas |
| `/reset` | POST | Reset environment (accepts `task_id`) |
| `/step` | POST | Execute an action |
| `/state` | GET | Get current episode state |
| `/ws` | WebSocket | Persistent session endpoint |
| `/docs` | GET | Interactive API documentation |
