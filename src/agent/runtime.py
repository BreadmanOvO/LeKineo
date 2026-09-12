"""Minimal Planner -> Executor -> Verifier -> Recovery runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .schemas import AgentState, TaskSpec
from .state import StateMachine
from .verifier import EnvironmentObservation, VerificationResult, Verifier
from .recovery import RecoveryPolicy


class TaskExecutor(Protocol):
    def reset(self, task: TaskSpec, attempt: int) -> None: ...

    def execute(self, task: TaskSpec, attempt: int) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RuntimeResult:
    status: str
    success: bool
    attempts: int
    retries: int
    failure_reason: str | None
    events: list[dict[str, Any]]
    verification: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "attempts": self.attempts,
            "retries": self.retries,
            "failure_reason": self.failure_reason,
            "events": self.events,
            "verification": self.verification,
        }


class AgentRuntime:
    def __init__(self, verifier: Verifier | None = None, recovery: RecoveryPolicy | None = None) -> None:
        self.verifier = verifier or Verifier()
        self.recovery = recovery or RecoveryPolicy(max_retries=1)

    def run(self, task: TaskSpec, executor: TaskExecutor) -> RuntimeResult:
        machine = StateMachine()
        machine.transition(AgentState.EXECUTING, reason="task_started", task_id=task.task_id)
        retries = 0
        attempts = 0
        last_result: VerificationResult | None = None
        while True:
            attempts += 1
            executor.reset(task, attempts)
            payload = executor.execute(task, attempts)
            observation = self._observation(task, payload)
            machine.transition(AgentState.VERIFYING, reason="rollout_complete", attempt=attempts)
            last_result = self.verifier.verify(observation, max_steps=task.max_steps)
            machine.events[-1]["data"]["verification"] = last_result.to_dict()
            if last_result.success:
                machine.transition(AgentState.SUCCEEDED, reason="verified_success", attempt=attempts)
                break
            decision = self.recovery.decide(last_result, retries_used=retries)
            if not decision.retryable:
                machine.transition(AgentState.FAILED, reason=decision.reason, attempt=attempts)
                break
            retries += 1
            machine.transition(AgentState.RETRYING, reason=decision.reason, retry=retries)
            machine.transition(AgentState.EXECUTING, reason="recovery_retry", retry=retries)
        assert last_result is not None
        return RuntimeResult(
            status="succeeded" if last_result.success else "failed",
            success=last_result.success,
            attempts=attempts,
            retries=retries,
            failure_reason=None if last_result.success else last_result.reason,
            events=machine.events,
            verification=last_result.to_dict(),
        )

    @staticmethod
    def _observation(task: TaskSpec, payload: dict[str, Any]) -> EnvironmentObservation:
        raw = payload.get("observation", payload)
        if isinstance(raw, EnvironmentObservation):
            return raw
        return EnvironmentObservation(task_id=task.task_id, **{key: raw[key] for key in EnvironmentObservation.__dataclass_fields__ if key != "task_id" and key in raw})
