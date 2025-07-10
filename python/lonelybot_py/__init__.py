"""Python helpers for the LonelyBot reinforcement learning interface."""

from .lonelybot_py import (
    GameState,
    MovePy,
    HeuristicConfigPy,
    ranked_moves_py as ranked_moves,
    best_move_py as best_move,
    best_move_mcts_py as best_move_mcts,
    column_probabilities_py as column_probabilities,
    analyze_state_py as analyze_state,
    collect_training_data_py as collect_training_data,
    generate_random_state_py as generate_random_state,
    step_py as _step,
    legal_actions_py as legal_actions,
    is_terminal_py as is_terminal,
    encode_observation_py as _encode_observation,
)

import numpy as np


def step(state: GameState, move: str):
    return _step(state, move)


def encode_observation(state: GameState) -> np.ndarray:
    data = _encode_observation(state)
    return np.array(data, dtype=np.int32)

__all__ = [
    "GameState",
    "MovePy",
    "HeuristicConfigPy",
    "ranked_moves",
    "best_move",
    "best_move_mcts",
    "column_probabilities",
    "analyze_state",
    "collect_training_data",
    "generate_random_state",
    "step",
    "legal_actions",
    "is_terminal",
    "encode_observation",
]
