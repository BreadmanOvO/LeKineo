"""Typed contracts for the Day 9 planner and task runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentState(str, Enum):
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    RETRYING = "RETRYING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


ALLOWED_EXECUTORS = {"libero_task_30", "libero_task_31", "libero_task_32", "libero_task_33", "libero_task_34", "libero_task_36", "libero_task_39"}


@dataclass(frozen=True)
class TaskSpec:
    executor: str
    task_id: int
    task_prompt: str
    max_steps: int = 400
    n_action_steps: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.executor not in ALLOWED_EXECUTORS:
            raise ValueError(f"unknown executor: {self.executor}")
        if self.task_id < 0 or self.max_steps <= 0 or self.n_action_steps <= 0:
            raise ValueError("task_id, max_steps and n_action_steps must be positive")
        if not self.task_prompt.strip():
            raise ValueError("task_prompt must not be empty")


@dataclass(frozen=True)
class TaskPlan:
    goal: str
    steps: tuple[TaskSpec, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [
                {
                    "executor": step.executor,
                    "task_id": step.task_id,
                    "task_prompt": step.task_prompt,
                    "max_steps": step.max_steps,
                    "n_action_steps": step.n_action_steps,
                    "seed": step.seed,
                }
                for step in self.steps
            ],
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SkillCall:
    executor: str
    task_id: int
    task_prompt: str
    arguments: dict[str, Any] = field(default_factory=dict)


def plan_to_skill_calls(plan: TaskPlan) -> list[SkillCall]:
    if not plan.steps:
        raise ValueError("plan must contain at least one step")
    return [SkillCall(step.executor, step.task_id, step.task_prompt) for step in plan.steps]
