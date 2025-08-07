import numpy as np
import torch
from torch import nn
from typing import Tuple

NB_ACTIONS = 256


class PolicyValueNet(nn.Module):
    """Simple policy-value network for Klondike."""

    def __init__(self, input_dim: int = 156, hidden_dim: int = 256, num_actions: int = NB_ACTIONS) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

        for layer in [self.fc1, self.fc2, self.policy_head, self.value_head]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute policy logits and value for a batch of states."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x)).squeeze(-1)
        return policy_logits, value

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return action probabilities and value for a single state."""
        self.eval()
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            policy_logits, value = self.forward(tensor)
            policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()[0]
            value = float(value.cpu().numpy()[0])
        return policy, value

    def save(self, path: str) -> None:
        """Save model parameters to file."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str) -> "PolicyValueNet":
        """Load model parameters from file and return model instance."""
        model = cls()
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model
