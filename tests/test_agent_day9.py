from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.agent import AgentState, Planner, StateMachine, TaskSpec, default_registry, plan_to_skill_calls


def test_planner_outputs_registered_json_plan():
    plan = Planner(default_registry()).plan("task 31: move bowl", executor="libero_task_31")
    assert plan.to_dict()["steps"][0]["executor"] == "libero_task_31"
    assert plan_to_skill_calls(plan)[0].task_id == 31


def test_unknown_executor_rejected():
    with pytest.raises(KeyError):
        Planner(default_registry()).plan("unknown", executor="libero_task_999")


def test_missing_prompt_rejected():
    with pytest.raises(ValueError):
        TaskSpec("libero_task_31", 31, "")


def test_state_machine_structured_events():
    machine = StateMachine()
    event = machine.transition(AgentState.EXECUTING, reason="test")
    assert event["from"] == "PLANNING"
    assert machine.state is AgentState.EXECUTING
