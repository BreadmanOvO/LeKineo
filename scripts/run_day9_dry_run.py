#!/usr/bin/env python3
"""Run the Day 9 planner/registry/state-machine dry run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import AgentState, DryRunExecutor, Planner, StateMachine, default_registry, plan_to_skill_calls  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", default="task 31: move the bowl to the plate")
    parser.add_argument("--executor", default="libero_task_31")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registry = default_registry()
    planner = Planner(registry)
    machine = StateMachine()
    machine.transition(AgentState.PLANNING, reason="runtime_start")
    plan = planner.plan(args.goal, executor=args.executor, seed=20260912)
    machine.transition(AgentState.EXECUTING, reason="plan_created", executor=args.executor)
    calls = plan_to_skill_calls(plan)
    skill_payload = registry.to_skill_call(calls[0])
    result = DryRunExecutor().execute(skill_payload)
    machine.transition(AgentState.VERIFYING, reason="dry_run_complete", actions_executed=result["actions_executed"])
    machine.transition(AgentState.SUCCEEDED, reason="dry_run_verified")
    payload = {"status": "PASS", "registered_tasks": registry.names(), "plan": plan.to_dict(), "skill_call": skill_payload, "result": result, "state": machine.state.value, "events": machine.events}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
