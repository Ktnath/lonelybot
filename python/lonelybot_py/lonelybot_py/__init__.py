"""Python helpers for the LonelyBot reinforcement learning interface."""

import numpy as np
from lonelybot_py.lonelybot_py import (
    GameState,
    MovePy,
    HeuristicConfigPy,
    ranked_moves_py,
    best_move_py,
    best_move_mcts_py,
    column_probabilities_py,
    analyze_state_py,
    collect_training_data_py,
    generate_random_state_py,
    step_py,
    legal_actions_py,
    is_terminal_py,
    encode_observation_py,
    reset_py,
    get_valid_actions_py,
    step_action_py,
    get_game_result_py,
    get_board_size_py,
    get_action_size_py,
    get_canonical_board_py,
)

def step(state: GameState, move: str):
    return step_py(state, move)

def reset():
    state, board = reset_py()
    return state, np.array(board, dtype=np.int8)

def step_action(state: GameState, action: int):
    next_state, board, reward, done = step_action_py(state, action)
    return next_state, np.array(board, dtype=np.int8), int(reward), bool(done)

def encode_observation(state: GameState) -> np.ndarray:
    data = encode_observation_py(state)
    return np.array(data, dtype=np.int16)

# Expose the functions with their public names
ranked_moves = ranked_moves_py
best_move = best_move_py
best_move_mcts = best_move_mcts_py
column_probabilities = column_probabilities_py
analyze_state = analyze_state_py
collect_training_data = collect_training_data_py
generate_random_state = generate_random_state_py
legal_actions = legal_actions_py
is_terminal = is_terminal_py
get_valid_actions = get_valid_actions_py
get_game_result = get_game_result_py
get_board_size = get_board_size_py
get_action_size = get_action_size_py
get_canonical_board = get_canonical_board_py

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
