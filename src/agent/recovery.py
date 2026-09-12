"""Bounded recovery decisions for task execution failures."""

from __future__ import annotations

from dataclasses import dataclass

from .verifier import VerificationResult


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    retryable: bool
    reason: str


class RecoveryPolicy:
    """Keep retries explicit and bounded; never retry an invisible target blindly."""

    def __init__(self, max_retries: int = 1) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.max_retries = max_retries

    def decide(self, result: VerificationResult, *, retries_used: int) -> RecoveryDecision:
        if result.success:
            return RecoveryDecision("finish", False, "task_completed")
        if result.reason == "target_not_visible":
            return RecoveryDecision("stop", False, result.reason)
        if retries_used >= self.max_retries:
            return RecoveryDecision("abort", False, "retry_budget_exhausted")
        if result.reason in {"collision", "environment_terminated", "timeout", "invalid_action"}:
            return RecoveryDecision("reset_and_retry", True, result.reason)
        return RecoveryDecision("abort", False, result.reason)
