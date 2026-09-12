"""Task registry and dry-run task executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .schemas import SkillCall, TaskSpec


@dataclass(frozen=True)
class RegisteredTask:
    spec: TaskSpec
    reset_kwargs: dict[str, Any]


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, RegisteredTask] = {}

    def register(self, spec: TaskSpec, **reset_kwargs: Any) -> None:
        if spec.executor in self._tasks:
            raise ValueError(f"executor already registered: {spec.executor}")
        self._tasks[spec.executor] = RegisteredTask(spec, reset_kwargs)

    def get(self, executor: str) -> RegisteredTask:
        try:
            return self._tasks[executor]
        except KeyError as exc:
            raise KeyError(f"unregistered executor: {executor}") from exc

    def to_skill_call(self, call: SkillCall) -> dict[str, Any]:
        registered = self.get(call.executor)
        if call.task_id != registered.spec.task_id:
            raise ValueError("task_id does not match registered executor")
        return {
            "executor": registered.spec.executor,
            "task_id": registered.spec.task_id,
            "task_prompt": registered.spec.task_prompt,
            "arguments": call.arguments,
            "reset_kwargs": registered.reset_kwargs,
        }

    def names(self) -> list[str]:
        return sorted(self._tasks)


class DryRunExecutor:
    """Adapter seam used before real LIBERO control is enabled."""

    def execute(self, skill_call: dict[str, Any], observation: Any = None, policy: Any = None) -> dict[str, Any]:
        return {
            "status": "dry_run",
            "executor": skill_call["executor"],
            "task_id": skill_call["task_id"],
            "actions_executed": 0,
            "policy_called": policy is not None,
        }


def default_registry() -> TaskRegistry:
    registry = TaskRegistry()
    registry.register(TaskSpec("libero_task_30", 30, "pick up the black bowl next to the cookie box and place it on the plate"), suite="libero_spatial")
    registry.register(TaskSpec("libero_task_31", 31, "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"), suite="libero_spatial")
    registry.register(TaskSpec("libero_task_32", 32, "pick up the black bowl on the ramekin and place it on the plate"), suite="libero_spatial")
    registry.register(TaskSpec("libero_task_33", 33, "pick up the black bowl on the stove and place it on the plate"), suite="libero_spatial")
    registry.register(TaskSpec("libero_task_34", 34, "pick up the black bowl between the plate and the ramekin and place it on the plate"), suite="libero_spatial")
    registry.register(TaskSpec("libero_task_36", 36, "pick up the black bowl next to the plate and place it on the plate"), suite="libero_spatial")
    registry.register(TaskSpec("libero_task_39", 39, "pick up the black bowl on the wooden cabinet and place it on the plate"), suite="libero_spatial")
    return registry
