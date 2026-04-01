"""
MLP scoring policy — main contribution.

score(x) = MLP(φ(x, state))

Small, shallow MLP (2 hidden layers, ReLU). The linear policy is a strict
special case; this captures nonlinear feature interactions.
"""

import numpy as np
import torch
import torch.nn as nn
from src.sat.state import SLSState
from src.policy.features import extract_batch, FEATURE_SETS


class MLPPolicy(nn.Module):
    def __init__(
        self,
        feature_set: str = "full",
        hidden_dim: int = 64,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.feature_set = feature_set
        n_features = len(FEATURE_SETS[feature_set])

        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def score(self, candidates: list[int], state: SLSState) -> np.ndarray:
        phi = extract_batch(candidates, state, self.feature_set)
        x = torch.from_numpy(phi)
        with torch.no_grad():
            scores = self.net(x).squeeze(-1)
        return scores.numpy()

    def score_tensor(self, candidates: list[int], state: SLSState) -> torch.Tensor:
        """Score with gradient tracking, for use during training."""
        phi = extract_batch(candidates, state, self.feature_set)
        x = torch.from_numpy(phi)
        return self.net(x).squeeze(-1)  # (n_cands,)

    def is_learnable(self) -> bool:
        return True
