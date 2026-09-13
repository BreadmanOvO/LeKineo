"""Real LIBERO executor adapter for Day 11 closed-loop evaluation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .schemas import TaskSpec


DATASET_TASK_TO_SPATIAL_INDEX = {
    30: 6, 31: 4, 32: 5, 33: 7, 34: 0, 35: 3, 36: 8, 37: 1, 38: 2, 39: 9,
}


class RealLiberoExecutor:
    """Execute one complete task rollout in the LeRobot LIBERO wrapper.

    The adapter deliberately keeps LIBERO-specific observation conversion here;
    the generic AgentRuntime only sees reset/execute and a state dictionary.
    """

    def __init__(self, policy: Any, *, suite_name: str = "libero_spatial", device: torch.device | None = None):
        self.policy = policy
        self.suite_name = suite_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = None
        self.pre = None
        self.post = None
        self.step_count = 0
        self.invalid_action = False
        self.inference_calls = 0
        self.total_environment_steps = 0
        self.inference_seconds = 0.0
        self.action_delta_sum = 0.0
        self.action_delta_count = 0
        self._previous_action: np.ndarray | None = None
        self._task: TaskSpec | None = None

    def reset(self, task: TaskSpec, attempt: int) -> None:
        from lerobot.envs.libero import LiberoEnv, _get_suite
        from lerobot.policies import make_pre_post_processors

        if self.env is not None:
            self.env.close()
        suite = _get_suite(self.suite_name)
        if task.task_id not in DATASET_TASK_TO_SPATIAL_INDEX:
            raise ValueError(f"no LIBERO spatial mapping for dataset task {task.task_id}")
        local_task_id = DATASET_TASK_TO_SPATIAL_INDEX[task.task_id]
        self.env = LiberoEnv(
            task_suite=suite,
            task_id=local_task_id,
            task_suite_name=self.suite_name,
            obs_type="pixels_agent_pos",
            episode_length=task.max_steps,
            init_states=True,
            control_mode="relative",
        )
        raw, _ = self.env.reset(seed=task.seed + attempt - 1)
        self._raw = raw
        self._task = task
        self.step_count = 0
        self.invalid_action = False
        self._previous_action = None
        if hasattr(self.policy, "config") and hasattr(self.policy.config, "n_action_steps"):
            self.policy.config.n_action_steps = int(task.n_action_steps)
        self.policy.reset()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            "lerobot/smolvla_base",
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    @staticmethod
    def _batch_state(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: RealLiberoExecutor._batch_state(item) for key, item in value.items()}
        return value.unsqueeze(0) if isinstance(value, torch.Tensor) and value.ndim > 0 else value

    def _observation(self, raw: dict[str, Any], task: TaskSpec) -> dict[str, Any]:
        from lerobot.envs.utils import preprocess_observation
        from lerobot.processor import LiberoProcessorStep

        obs = preprocess_observation(raw)
        obs["observation.robot_state"] = self._batch_state(obs["observation.robot_state"])
        obs["task"] = [task.task_prompt]
        obs = LiberoProcessorStep().observation(obs)
        return {
            "observation.images.camera1": obs["observation.images.image"],
            "observation.images.camera2": obs["observation.images.image2"],
            "observation.images.camera3": obs["observation.images.image2"],
            "observation.state": obs["observation.state"][..., :6],
            "task": obs["task"],
        }

    def execute(self, task: TaskSpec, attempt: int) -> dict[str, Any]:
        if self.env is None or self._task is None:
            raise RuntimeError("reset must be called before execute")
        terminated = truncated = False
        reward = 0.0
        info: dict[str, Any] = {}
        started = time.perf_counter()
        for _ in range(task.max_steps):
            obs = self._observation(self._raw, task)
            with torch.inference_mode():
                queues = getattr(self.policy, "_queues", {})
                action_queue = queues.get("action")
                generates_chunk = action_queue is None or len(action_queue) == 0
                if generates_chunk:
                    self.inference_calls += 1
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    inference_started = time.perf_counter()
                action6 = self.post(self.policy.select_action(self.pre(obs)))
                if generates_chunk:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    self.inference_seconds += time.perf_counter() - inference_started
            action6 = action6.detach().float().cpu().numpy().reshape(-1)
            if action6.shape[0] < 6 or not np.isfinite(action6[:6]).all():
                self.invalid_action = True
                break
            action7 = np.zeros(7, dtype=np.float32)
            action7[:6] = action6[:6]
            if self._previous_action is not None:
                self.action_delta_sum += float(np.abs(action7 - self._previous_action).mean())
                self.action_delta_count += 1
            self._previous_action = action7.copy()
            self._raw, reward, terminated, truncated, info = self.env.step(action7)
            self.step_count += 1
            self.total_environment_steps += 1
            if terminated or truncated:
                break
        rollout_step_ms = 1000.0 * (time.perf_counter() - started) / max(1, self.step_count)
        success = bool(info.get("success", False) or (terminated and float(reward) > 0.0))
        return {
            "step_count": self.step_count,
            "success": success,
            "object_in_target": success,
            "terminated": bool(terminated or truncated),
            "invalid_action": self.invalid_action,
            "mean_inference_ms": 1000.0 * self.inference_seconds / max(1, self.inference_calls),
            "mean_rollout_step_ms": rollout_step_ms,
            "inference_calls": self.inference_calls,
            "total_environment_steps": self.total_environment_steps,
            "action_smoothness_l1": self.action_delta_sum / max(1, self.action_delta_count),
            "n_action_steps": task.n_action_steps,
            "reward": float(reward),
            "info": {str(key): str(value) for key, value in info.items()},
        }

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
