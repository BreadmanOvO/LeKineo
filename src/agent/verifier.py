"""Environment-state based task verification for the Day 10 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnvironmentObservation:
    """Minimal state contract extracted from a LIBERO/MuJoCo environment."""

    task_id: int
    object_visible: bool = True
    object_grasped: bool = False
    object_in_target: bool = False
    collision: bool = False
    terminated: bool = False
    success: bool = False
    step_count: int = 0
    invalid_action: bool = False
    inference_calls: int = 0
    mean_inference_ms: float = 0.0


@dataclass(frozen=True)
class VerificationResult:
    status: str
    success: bool
    reason: str
    recovery: str
    confidence: float
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "reason": self.reason,
            "recovery": self.recovery,
            "confidence": self.confidence,
            "details": self.details,
        }


class Verifier:
    """Verify task state without trusting the policy's self-reported success."""

    def verify(self, observation: EnvironmentObservation, *, max_steps: int) -> VerificationResult:
        if observation.invalid_action:
            return self._failure("invalid_action", "retry_or_abort", observation)
        if not observation.object_visible:
            return self._failure("target_not_visible", "stop", observation)
        if observation.success or observation.object_in_target:
            return VerificationResult("succeeded", True, "task_completed", "none", 0.99, self._details(observation))
        if observation.collision:
            return self._failure("collision", "reset_and_retry", observation)
        if observation.terminated:
            return self._failure("environment_terminated", "reset_and_retry", observation)
        if observation.step_count >= max_steps:
            return self._failure("timeout", "reset_and_retry", observation)
        return VerificationResult("in_progress", False, "not_completed", "continue", 0.50, self._details(observation))

    @staticmethod
    def _details(observation: EnvironmentObservation) -> dict[str, Any]:
        return {
            "task_id": observation.task_id,
            "object_visible": observation.object_visible,
            "object_grasped": observation.object_grasped,
            "object_in_target": observation.object_in_target,
            "collision": observation.collision,
            "terminated": observation.terminated,
            "step_count": observation.step_count,
            "invalid_action": observation.invalid_action,
            "inference_calls": observation.inference_calls,
            "mean_inference_ms": observation.mean_inference_ms,
        }

    def _failure(self, reason: str, recovery: str, observation: EnvironmentObservation) -> VerificationResult:
        return VerificationResult("failed", False, reason, recovery, 0.95, self._details(observation))
