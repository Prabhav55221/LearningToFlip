"""
MLP scoring policy — main contribution.

score(x) = MLP(φ(x, state))

Hidden dim and depth are configurable; default (64, 2) for full experiments,
with smaller variants used for controlled ablations when needed.

noise_prob: fixed random-walk probability matching Interian's escape mechanism.
  - At inference: with prob noise_prob pick a random candidate (by_policy=False)
  - At training: noise steps fire in the training loop but do NOT push to the
    REINFORCE buffer — gradient only flows through policy-selected actions.
  - Not learned; set per family from the Interian p_dict values.

normalize: divide count features by avg_deg and time features by step —
  makes the policy scale-invariant across instance sizes (critical for generalization).
"""

import math
import random
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
        normalize: bool = False,
        noise_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_set = feature_set
        self.normalize   = normalize
        self.noise_prob  = noise_prob   # fixed, not learned
        n_features = len(FEATURE_SETS[feature_set])

        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def select(self, candidates: list[int], state: SLSState) -> tuple[int, float, bool]:
        """Inference: softmax sample with optional fixed noise walk."""
        k   = len(candidates)
        phi = extract_batch(candidates, state, self.feature_set, normalize=self.normalize)

        # Fixed noise walk — random escape, no gradient involvement
        if self.noise_prob > 0.0 and random.random() < self.noise_prob:
            idx = random.randint(0, k - 1)
            return candidates[idx], math.log(1.0 / k), False

        x = torch.from_numpy(phi)
        with torch.no_grad():
            scores = self.net(x).squeeze(-1)
            probs  = torch.softmax(scores, dim=0)
        idx      = int(torch.multinomial(probs, 1).item())
        log_prob = float(torch.log(probs[idx]).item())
        return candidates[idx], log_prob, True

    def log_prob_phi(self, phi: np.ndarray) -> torch.Tensor:
        """Log-probabilities from pre-extracted features (with gradient). Used by REINFORCE trainer."""
        scores = self.net(torch.from_numpy(phi).float()).squeeze(-1)
        return torch.log_softmax(scores, dim=0)

    def score_tensor(self, candidates: list[int], state: SLSState) -> torch.Tensor:
        phi = extract_batch(candidates, state, self.feature_set, normalize=self.normalize)
        return self.net(torch.from_numpy(phi)).squeeze(-1)

    def score_logits(self, candidates: list[int], state: SLSState) -> torch.Tensor:
        return self.score_tensor(candidates, state)

    def score_phi(self, phi: np.ndarray) -> torch.Tensor:
        return self.net(torch.from_numpy(phi).float()).squeeze(-1)

    def is_learnable(self) -> bool:
        return True
