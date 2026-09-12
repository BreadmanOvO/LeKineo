"""Deterministic task planner for registered LIBERO executors."""

from __future__ import annotations

import re

from .schemas import TaskPlan, TaskSpec
from .skills import TaskRegistry


class Planner:
    def __init__(self, registry: TaskRegistry) -> None:
        self.registry = registry

    def plan(self, goal: str, *, executor: str | None = None, seed: int = 0) -> TaskPlan:
        if not goal.strip():
            raise ValueError("goal must not be empty")
        if executor is None:
            match = re.search(r"(?:task|任务)[ _-]?(\d+)", goal.lower())
            if not match:
                raise ValueError("executor is required when goal does not contain a task id")
            executor = f"libero_task_{match.group(1)}"
        registered = self.registry.get(executor)
        spec = registered.spec
        step = TaskSpec(spec.executor, spec.task_id, spec.task_prompt, spec.max_steps, spec.n_action_steps, seed)
        return TaskPlan(goal=goal, steps=(step,), metadata={"planner": "deterministic_day9", "registered_executor": executor})
