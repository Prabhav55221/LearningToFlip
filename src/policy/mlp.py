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

    def select(self, candidates: list[int], state: SLSState) -> tuple[int, float, bool]:
        """Inference: softmax sample, no gradient."""
        phi = extract_batch(candidates, state, self.feature_set)
        x = torch.from_numpy(phi)
        with torch.no_grad():
            scores = self.net(x).squeeze(-1)
        probs = torch.softmax(scores, dim=0)
        idx = int(torch.multinomial(probs, num_samples=1).item())
        log_prob = float(torch.log(probs[idx]).item())
        return candidates[idx], log_prob, True

    def score_tensor(self, candidates: list[int], state: SLSState) -> torch.Tensor:
        """Training: scores with gradient tracking for REINFORCE."""
        phi = extract_batch(candidates, state, self.feature_set)
        x = torch.from_numpy(phi)
        return self.net(x).squeeze(-1)  # shape (n_candidates,)

    def score_logits(self, candidates: list[int], state: SLSState) -> torch.Tensor:
        return self.score_tensor(candidates, state)

    def score_phi(self, phi: np.ndarray) -> torch.Tensor:
        """Score from a pre-extracted feature matrix (k, n_features). Used by REINFORCE trainer."""
        return self.net(torch.from_numpy(phi).float()).squeeze(-1)

    def is_learnable(self) -> bool:
        return True
