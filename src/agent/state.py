"""State machine and structured transition events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .schemas import AgentState


@dataclass
class StateMachine:
    state: AgentState = AgentState.PLANNING
    events: list[dict[str, Any]] = field(default_factory=list)

    def transition(self, new_state: AgentState, *, reason: str = "", **data: Any) -> dict[str, Any]:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from": self.state.value,
            "to": new_state.value,
            "reason": reason,
            "data": data,
        }
        self.state = new_state
        self.events.append(event)
        return event
