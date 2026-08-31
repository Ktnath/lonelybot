import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

from lonelybot_py import (
    get_action_size_py,
    get_board_size_py,
    get_valid_actions_py,
    reset_py,
    step_action_py,
)


class KlondikeEnv(gym.Env):
    """Gymnasium adapter around the current ``lonelybot_py`` bindings.

    The Rust binding owns the game state. Observations are flattened to a
    100-element float32 vector and legal action indices live in [0, 214].
    """

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        board_shape = tuple(int(v) for v in get_board_size_py())
        self._obs_size = int(np.prod(board_shape))
        self._action_size = int(get_action_size_py())

        self.action_space = spaces.Discrete(self._action_size)
        self.observation_space = spaces.Box(
            low=0.0,
            high=255.0,
            shape=(self._obs_size,),
            dtype=np.float32,
        )
        self.state: Optional[Any] = None
        self._obs = np.zeros(self._obs_size, dtype=np.float32)

    @staticmethod
    def _flatten_board(board: Any) -> np.ndarray:
        return np.asarray(board, dtype=np.float32).reshape(-1)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new game.

        ``seed`` currently seeds Gymnasium's RNG only; deterministic Rust-side
        deal selection is a separate research task and is reported in ``info``.
        """
        super().reset(seed=seed)
        self.state, board = reset_py()
        self._obs = self._flatten_board(board)
        info: Dict[str, Any] = {
            "valid_actions": list(get_valid_actions_py(self.state)),
            "rust_seeded_reset": False,
        }
        return self._obs.copy(), info

    def action_mask(self) -> np.ndarray:
        """Return a boolean mask of legal actions for the current state."""
        if self.state is None:
            raise RuntimeError("Call reset() before action_mask().")
        mask = np.zeros(self._action_size, dtype=bool)
        mask[np.asarray(get_valid_actions_py(self.state), dtype=np.int64)] = True
        return mask

    def step(self, action: int):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        valid_actions = set(int(a) for a in get_valid_actions_py(self.state))
        action = int(action)
        if action not in valid_actions:
            info = {"legal": False, "valid_actions": sorted(valid_actions)}
            return self._obs.copy(), -1.0, False, False, info

        self.state, board, reward, done = step_action_py(self.state, action)
        self._obs = self._flatten_board(board)
        next_valid = list(get_valid_actions_py(self.state)) if not done else []
        info = {
            "legal": True,
            "valid_actions": next_valid,
        }
        return self._obs.copy(), float(reward), bool(done), False, info

    def render(self):  # pragma: no cover - rendering not implemented
        return None
