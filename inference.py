"""Inference script for Incident Response Environment.

Uses the OpenAI-compatible client to run an LLM agent against all 4 tasks
and produce reproducible baseline scores.

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key

Usage:
    # Start the environment server first:
    uvicorn incident_response_env.server.app:app --port 8000

    # Then run inference:
    API_BASE_URL=https://router.huggingface.co/v1 \
    MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
    HF_TOKEN=hf_... \
    python inference.py
"""

import json
import os
import re
import sys

from openai import OpenAI

# Ensure imports work from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from incident_response_env import IncidentResponseEnv, IncidentResponseAction
from incident_response_env.scenarios import TASK_IDS

# ---------------------------------------------------------------------------
# Required environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_STEPS = 25  # Hard cap per episode to stay within 20min total runtime
TEMPERATURE = 0.0
MAX_TOKENS = 300

SYSTEM_PROMPT = """\
You are an experienced Site Reliability Engineer (SRE) on-call. You've received an infrastructure alert and must diagnose and resolve the incident.

## Your Workflow
1. INVESTIGATE: Check metrics and logs of services to understand the situation
2. DIAGNOSE: Identify the root cause by reasoning about the evidence
3. REMEDIATE: Take the correct action to fix the issue

## Available Commands
Investigation:
- check_metrics <service>  — View current metrics (CPU, memory, disk, error rates, etc.)
- read_logs <service>      — Read recent application/system logs
- check_status <service>   — Check health status, uptime, deploy history

Remediation:
- restart_service <service>    — Restart a service
- scale_service <service>      — Scale up service replicas
- rollback_deploy <service>    — Roll back to previous deployment version
- clear_disk <service>         — Clean up disk space (WAL, temp files, etc.)

Other:
- diagnose                     — Submit your root cause analysis
- escalate                     — Escalate to a senior engineer (use only if truly stuck)

## Available Services
api-gateway, user-service, order-service, postgres-primary, redis-cache, kafka-broker

## Response Format
Respond with ONLY a JSON object on a single line, no markdown:
{"command": "<command>", "target": "<service>", "parameters": {}}

For the diagnose command, include an explanation:
{"command": "diagnose", "target": "", "parameters": {"explanation": "Root cause is..."}}

## Strategy
- Start by checking metrics and logs of the service mentioned in the alert
- Follow the evidence to find the root cause (it may be in a different service)
- Consider cascading failures: the alerting service may just be a symptom
- Submit a diagnosis before remediating when possible
- Be efficient: fewer steps is better
"""


def parse_action(response_text: str) -> IncidentResponseAction:
    """Parse LLM response into an IncidentResponseAction."""
    text = response_text.strip()

    # Handle markdown code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    # Try direct JSON parse
    try:
        data = json.loads(text)
        return IncidentResponseAction(
            command=data.get("command", ""),
            target=data.get("target", ""),
            parameters=data.get("parameters", {}),
        )
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON object in the text
    json_match = re.search(r"\{[^{}]*\}", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return IncidentResponseAction(
                command=data.get("command", ""),
                target=data.get("target", ""),
                parameters=data.get("parameters", {}),
            )
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: try to parse as simple command
    parts = text.split()
    if parts:
        return IncidentResponseAction(
            command=parts[0],
            target=parts[1] if len(parts) > 1 else "",
        )

    return IncidentResponseAction(command="escalate")


def run_task(client: OpenAI, env_url: str, task_id: str, verbose: bool = True) -> float:
    """Run a single task and return the score."""
    with IncidentResponseEnv(base_url=env_url).sync() as env:
        result = env.reset(task_id=task_id)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"ALERT:\n{result.observation.alert_summary}\n\n"
                    f"Available services: {', '.join(result.observation.available_services)}\n"
                    f"Available commands: {', '.join(result.observation.available_commands)}\n\n"
                    f"Begin your investigation."
                ),
            },
        ]

        step = 0
        while not result.done and step < MAX_STEPS:
            step += 1

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                assistant_msg = response.choices[0].message.content or ""
            except Exception as exc:
                print(f"  LLM request failed ({exc}). Using escalate fallback.")
                assistant_msg = '{"command": "escalate", "target": "", "parameters": {}}'

            messages.append({"role": "assistant", "content": assistant_msg})

            action = parse_action(assistant_msg)

            if verbose:
                print(f"  Step {step}: {action.command} {action.target}")

            result = env.step(action)

            # Feed observation back to the LLM
            feedback = (
                f"Command output:\n{result.observation.command_output}\n\n"
                f"Reward: {result.reward} | "
                f"Services investigated: {result.observation.services_investigated} | "
                f"Time elapsed: {result.observation.time_elapsed_minutes}min"
            )

            if result.done:
                feedback += "\n\nEpisode complete."

            messages.append({"role": "user", "content": feedback})

        score = result.reward if result.reward is not None else 0.0
        state = env.state()

        if verbose:
            print(f"  Result: score={score:.4f}, steps={state.step_count}")

    return score


def main() -> None:
    """Run inference on all tasks."""
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable is required.")
        print("Usage: MODEL_NAME=meta-llama/... HF_TOKEN=hf_... python inference.py")
        sys.exit(1)

    if not API_KEY:
        print("ERROR: HF_TOKEN environment variable is required.")
        sys.exit(1)

    # Create OpenAI-compatible client with required env vars
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env_url = os.getenv("ENV_URL", "http://localhost:8000")
    verbose = os.getenv("QUIET", "").lower() not in ("1", "true", "yes")

    print("Incident Response Environment — Inference")
    print(f"  Model:       {MODEL_NAME}")
    print(f"  API URL:     {API_BASE_URL}")
    print(f"  Environment: {env_url}")
    print(f"  Tasks:       {TASK_IDS}")
    print("=" * 60)

    scores = {}
    for task_id in TASK_IDS:
        print(f"\n--- Task: {task_id} ---")
        try:
            score = run_task(client, env_url, task_id, verbose=verbose)
            scores[task_id] = score
        except Exception as e:
            print(f"  ERROR: {e}")
            scores[task_id] = 0.0

    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    for task_id, score in scores.items():
        difficulty = {
            "disk_full": "easy",
            "bad_deploy": "medium",
            "memory_and_cache": "hard",
            "kafka_lag": "medium-hard",
        }.get(task_id, "unknown")
        print(f"  {task_id:25s} ({difficulty:12s}): {score:.4f}")

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  {'Average':25s}              : {avg:.4f}")


if __name__ == "__main__":
    main()
