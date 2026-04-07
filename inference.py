"""Inference script for Incident Response Environment.

Uses the OpenAI-compatible client to run an LLM agent against all 4 tasks
and produce reproducible baseline scores.

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key

STDOUT FORMAT:
    [START] task=<task_name> env=incident_response model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Usage:
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

# Import from flat repo structure (all .py files at root level)
try:
    from incident_response_env import IncidentResponseEnv, IncidentResponseAction
    from incident_response_env.scenarios import TASK_IDS
except ModuleNotFoundError:
    from client import IncidentResponseEnv
    from models import IncidentResponseAction
    from scenarios import TASK_IDS

# ---------------------------------------------------------------------------
# Required environment variables
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BENCHMARK = "incident_response"
MAX_STEPS = 25
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


def run_task(client: OpenAI, env_url: str, task_id: str) -> float:
    """Run a single task and return the score."""
    step = 0
    step_rewards = []
    score = 0.0

    # [START] — always emitted at episode begin
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")

    try:
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
                    assistant_msg = '{"command": "escalate", "target": "", "parameters": {}}'

                messages.append({"role": "assistant", "content": assistant_msg})

                action = parse_action(assistant_msg)
                action_str = f"{action.command}({action.target})" if action.target else f"{action.command}()"

                result = env.step(action)

                reward = result.reward if result.reward is not None else 0.0
                step_rewards.append(reward)
                done_str = "true" if result.done else "false"

                # [STEP] — emitted immediately after env.step()
                print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error=null")

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

    except Exception as e:
        # Ensure [END] is emitted even on exception
        error_msg = str(e).replace('\n', ' ').replace('\r', '')
        print(f"[STEP] step={step + 1} action=error() reward=0.00 done=true error={error_msg}", flush=True)
        step_rewards.append(0.0)
        step += 1

    # [END] — always emitted, even on exception
    rewards_str = ",".join(f"{r:.2f}" for r in step_rewards) if step_rewards else "0.00"
    success_str = "true" if score > 0.5 else "false"
    print(f"[END] success={success_str} steps={step} rewards={rewards_str}")

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

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_url = os.getenv("ENV_URL", "http://localhost:8000")

    scores = {}
    for task_id in TASK_IDS:
        try:
            score = run_task(client, env_url, task_id)
            scores[task_id] = score
        except Exception as e:
            print(f"[END] success=false steps=0 rewards=0.00")
            scores[task_id] = 0.0

    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\nAverage score: {avg:.4f}")


if __name__ == "__main__":
    main()
