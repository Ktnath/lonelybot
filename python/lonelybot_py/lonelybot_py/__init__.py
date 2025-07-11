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
    reset_py as _reset,
    get_valid_actions_py as get_valid_actions,
    step_action_py as _step_action,
    get_game_result_py as get_game_result,
    get_board_size_py as get_board_size,
    get_action_size_py as get_action_size,
    get_canonical_board_py as get_canonical_board,
)

import numpy as np


def step(state: GameState, move: str):
    return _step(state, move)


def reset():
    state, board = _reset()
    return state, np.array(board, dtype=np.int8)


def step_action(state: GameState, action: int):
    next_state, board, reward, done = _step_action(state, action)
    return next_state, np.array(board, dtype=np.int8), int(reward), bool(done)


def encode_observation(state: GameState) -> np.ndarray:
    data = _encode_observation(state)
    return np.array(data, dtype=np.int16)

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
    "step_action",
    "reset",
    "legal_actions",
    "is_terminal",
    "get_valid_actions",
    "get_game_result",
    "get_board_size",
    "get_action_size",
    "get_canonical_board",
    "encode_observation",
]
