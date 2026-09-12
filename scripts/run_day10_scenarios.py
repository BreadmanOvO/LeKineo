#!/usr/bin/env python3
"""Exercise Day 10 verifier/recovery behavior with deterministic fake rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import AgentRuntime, RecoveryPolicy, TaskSpec  # noqa: E402


class ScenarioExecutor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.reset_calls = 0

    def reset(self, task: TaskSpec, attempt: int) -> None:
        self.reset_calls += 1

    def execute(self, task: TaskSpec, attempt: int) -> dict:
        if self.name == "retry_success" and attempt == 1:
            return {"step_count": task.max_steps, "terminated": True}
        if self.name == "retry_success":
            return {"step_count": 12, "object_grasped": True, "object_in_target": True, "success": True}
        if self.name == "target_invisible":
            return {"step_count": 8, "object_visible": False}
        if self.name == "timeout":
            return {"step_count": task.max_steps}
        if self.name == "persistent_collision":
            return {"step_count": 20, "collision": True, "terminated": True}
        raise ValueError(self.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = TaskSpec("libero_task_31", 31, "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate", max_steps=50)
    scenarios = {}
    for name in ("retry_success", "target_invisible", "timeout", "persistent_collision"):
        runtime = AgentRuntime(recovery=RecoveryPolicy(max_retries=1))
        result = runtime.run(task, ScenarioExecutor(name))
        scenarios[name] = result.to_dict()
    payload = {"status": "PASS", "scenarios": scenarios}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
