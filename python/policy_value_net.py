import numpy as np
import torch
from torch import nn
from typing import Optional, Tuple, Union

BOARD_SIZE = 100
NB_ACTIONS = 215
DeviceLike = Union[str, torch.device]


class PolicyValueNet(nn.Module):
    """Policy/value network for Klondike with CPU/GPU batch inference support."""

    def __init__(
        self,
        input_dim: int = BOARD_SIZE,
        hidden_dim: int = 256,
        num_actions: int = NB_ACTIONS,
        device: Optional[DeviceLike] = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        for layer in [self.fc1, self.fc2, self.policy_head, self.value_head]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute policy logits and scalar value for a batch of states."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x)).squeeze(-1)
        return policy_logits, value

    @torch.inference_mode()
    def predict_batch(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return policy probabilities and values for one or more states."""
        self.eval()
        tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2 or tensor.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected shape (batch, {self.input_dim}) or ({self.input_dim},), got {tuple(tensor.shape)}"
            )

        policy_logits, value = self.forward(tensor)
        policy = torch.softmax(policy_logits, dim=-1)
        return policy.cpu().numpy(), value.cpu().numpy()

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return action probabilities and value for a single state."""
        policy, value = self.predict_batch(x)
        return policy[0], float(value[0])

    def save(self, path: str) -> None:
        """Save weights together with the architecture metadata."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_actions": self.num_actions,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: Optional[DeviceLike] = None) -> "PolicyValueNet":
        """Load a checkpoint, including legacy state-dict-only checkpoints."""
        map_location = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        payload = torch.load(path, map_location=map_location)

        if isinstance(payload, dict) and "state_dict" in payload:
            model = cls(
                input_dim=int(payload.get("input_dim", BOARD_SIZE)),
                hidden_dim=int(payload.get("hidden_dim", 256)),
                num_actions=int(payload.get("num_actions", NB_ACTIONS)),
                device=map_location,
            )
            state_dict = payload["state_dict"]
        else:
            model = cls(device=map_location)
            state_dict = payload

        model.load_state_dict(state_dict)
        model.eval()
        return model
