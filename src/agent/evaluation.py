"""Shared episode-level accounting for closed-loop evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EpisodeMetrics:
    system: str
    checkpoint: str
    task_suite: str
    task_id: int
    split: str
    seed: int
    episode: int
    success: bool
    episode_steps: int
    early_termination: bool
    invalid_action: bool
    retries: int
    recovery_success: bool
    n_action_steps: int
    mean_inference_ms: float
    peak_vram_gb: float
    status: str
    failure_reason: str | None

    def to_row(self) -> dict[str, Any]:
        return self.__dict__.copy()
