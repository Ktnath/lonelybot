"""Common interfaces for experimental agents."""

from __future__ import annotations

from typing import Protocol, Any


class Agent(Protocol):
    name: str

    def choose_action(self, state: Any) -> Any:
        """Return the action selected for the supplied state."""
        ...
