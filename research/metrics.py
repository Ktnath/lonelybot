"""Metrics utilities for Lonelybot research experiments."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class EpisodeMetrics:
    seed: int
    agent: str
    draw_step: int
    won: bool
    moves: int
    decision_time_s: float
    terminal_reason: str = ""
    oracle_moves: Optional[int] = None
    oracle_regret: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
