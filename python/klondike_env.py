import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple

from klondike_core import (
    new_game,
    legal_moves,
    do_move,
    is_win,
    encode_observation_py,
    move_to_action_idx,
    get_valid_actions_py,
    NB_ACTIONS,
)


class KlondikeEnv(gym.Env):
    """Gymnasium environment for the Klondike solitaire engine."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Discrete(NB_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=(156,), dtype=np.int32)
        self.state: Optional[str] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Start a new game and return the initial observation."""
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**32 - 1))
        self.state = new_game(int(seed))
        obs = np.array(encode_observation_py(self.state), dtype=np.int32)
        return obs, {}

    def step(self, action: int):
        assert self.state is not None, "Call reset() before step()."

        legal_move_strings = legal_moves(self.state)
        action_map = {move_to_action_idx(m): m for m in legal_move_strings}

        if action in action_map:
            mv = action_map[action]
            self.state = do_move(self.state, mv)
            legal = True
            done = is_win(self.state)
            if done:
                reward = 100
            else:
                remaining = get_valid_actions_py(self.state)
                if not remaining:
                    done = True
                    reward = -1
                else:
                    reward = 1
        else:
            mv = None
            legal = False
            done = False
            reward = -1

        obs = np.array(encode_observation_py(self.state), dtype=np.int32)
        info = {"move": mv, "legal": legal}
        return obs, reward, done, False, info

    def render(self):  # pragma: no cover - rendering not implemented
        pass
