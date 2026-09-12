from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import AgentRuntime, RecoveryPolicy, TaskSpec


TASK = TaskSpec("libero_task_31", 31, "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate", max_steps=20)


class FakeExecutor:
    def __init__(self, payloads):
        self.payloads = payloads
        self.attempt = 0
        self.reset_calls = 0

    def reset(self, task, attempt):
        self.reset_calls += 1

    def execute(self, task, attempt):
        self.attempt = attempt
        return self.payloads[attempt - 1]


def test_failed_rollout_retries_and_succeeds():
    executor = FakeExecutor([{"terminated": True, "step_count": 20}, {"success": True, "object_in_target": True}])
    result = AgentRuntime(recovery=RecoveryPolicy(1)).run(TASK, executor)
    assert result.success is True
    assert result.attempts == 2
    assert result.retries == 1
    assert executor.reset_calls == 2


def test_invisible_target_stops_without_retry():
    executor = FakeExecutor([{"object_visible": False, "step_count": 3}])
    result = AgentRuntime(recovery=RecoveryPolicy(1)).run(TASK, executor)
    assert result.success is False
    assert result.failure_reason == "target_not_visible"
    assert result.attempts == 1


def test_timeout_retries_then_aborts():
    executor = FakeExecutor([{"step_count": 20}, {"step_count": 20}])
    result = AgentRuntime(recovery=RecoveryPolicy(1)).run(TASK, executor)
    assert result.failure_reason == "timeout"
    assert result.attempts == 2
    assert result.retries == 1


def test_persistent_collision_has_bounded_exit():
    executor = FakeExecutor([{"collision": True}, {"collision": True}])
    result = AgentRuntime(recovery=RecoveryPolicy(1)).run(TASK, executor)
    assert result.success is False
    assert result.attempts == 2
    assert result.events[-1]["to"] == "FAILED"
