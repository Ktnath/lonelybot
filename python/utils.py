"""Utility helpers for JSON loading."""

from typing import List, Optional

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
SUITS = ["H", "D", "C", "S"]


def parse_hidden(values: List):
    """Convert JSON values to optional card strings.

    ``"unknown"`` or ``-1`` are translated to ``None``.
    """
    result: List[Optional[str]] = []
    for v in values:
        if v == "unknown" or v == -1:
            result.append(None)
        else:
            result.append(str(v))
    return result
