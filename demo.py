"""Demo script for the Incident Response Environment.

Shows a complete walkthrough of the 'disk_full' scenario — the easiest task —
where we investigate postgres-primary and clear disk to resolve the alert.

Usage:
    # Against your deployed HF Space:
    python demo.py https://rajasuryarjs-test123.hf.space

    # Against a local server:
    python demo.py http://localhost:8000
"""

import os
import sys

# Allow running from inside the incident_response_env directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from incident_response_env import IncidentResponseEnv, IncidentResponseAction


def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    print(f"Connecting to {base_url} ...")

    with IncidentResponseEnv(base_url=base_url).sync() as env:
        # --- Reset to the disk_full scenario ---
        result = env.reset(task_id="disk_full")
        obs = result.observation
        print(f"\n{'='*60}")
        print(f"Task: disk_full (Easy)")
        print(f"Alert: {obs.alert_summary}")
        print(f"Severity: {obs.severity}")
        print(f"Available services: {obs.available_services}")
        print(f"{'='*60}\n")

        # --- Step 1: Check metrics on postgres-primary ---
        print("Step 1: Checking metrics on postgres-primary ...")
        result = env.step(IncidentResponseAction(
            command="check_metrics",
            target="postgres-primary",
        ))
        obs = result.observation
        print(f"  Output: {obs.command_output[:200]}")
        print(f"  Reward so far: {result.reward}")

        # --- Step 2: Read logs on postgres-primary ---
        print("\nStep 2: Reading logs on postgres-primary ...")
        result = env.step(IncidentResponseAction(
            command="read_logs",
            target="postgres-primary",
        ))
        obs = result.observation
        print(f"  Output: {obs.command_output[:200]}")
        print(f"  Reward so far: {result.reward}")

        # --- Step 3: Diagnose the root cause ---
        print("\nStep 3: Diagnosing root cause ...")
        result = env.step(IncidentResponseAction(
            command="diagnose",
            target="postgres-primary",
            parameters={"explanation": "Disk full due to WAL segments not being archived and temp table bloat"},
        ))
        obs = result.observation
        print(f"  Output: {obs.command_output[:200]}")
        print(f"  Reward so far: {result.reward}")

        # --- Step 4: Clear disk to remediate ---
        print("\nStep 4: Clearing disk on postgres-primary ...")
        result = env.step(IncidentResponseAction(
            command="clear_disk",
            target="postgres-primary",
        ))
        obs = result.observation
        print(f"  Output: {obs.command_output[:200]}")
        print(f"  Reward so far: {result.reward}")
        print(f"  Done: {result.done}")

        # --- Summary ---
        print(f"\n{'='*60}")
        print(f"Episode complete!")
        print(f"  Final reward: {result.reward}")
        print(f"  Steps taken: {len(obs.actions_taken)}")
        print(f"  Services investigated: {obs.services_investigated}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
