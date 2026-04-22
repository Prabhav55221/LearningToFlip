"""
MLP scoring policy — main contribution.

score(x) = MLP(φ(x, state))

Small, shallow MLP (2 hidden layers, ReLU). The linear policy is a strict
special case; this captures nonlinear feature interactions.

noise_walk=True adds a learned noise-walk probability p_w = 0.5·sigmoid(w),
matching Interian's escape mechanism: with probability p_w a random variable
is chosen; otherwise the MLP softmax is used. Training uses the mixture
log-prob so the policy learns to calibrate p_w jointly with the MLP weights.
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
        noise_walk: bool = False,
    ) -> None:
        super().__init__()
        self.feature_set = feature_set
        self.normalize   = normalize
        self.noise_walk  = noise_walk
        n_features = len(FEATURE_SETS[feature_set])

        layers: list[nn.Module] = []
        in_dim = n_features
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        if noise_walk:
            # p_w = 0.5 * sigmoid(noise_param); init → p_w ≈ 0.25
            self.noise_param = nn.Parameter(torch.zeros(1))

    @property
    def noise_prob(self) -> float:
        if not self.noise_walk:
            return 0.0
        return float(0.5 * torch.sigmoid(self.noise_param).item())

    def select(self, candidates: list[int], state: SLSState) -> tuple[int, float, bool]:
        """Inference: mixture-aware softmax sample, no gradient."""
        k   = len(candidates)
        phi = extract_batch(candidates, state, self.feature_set, normalize=self.normalize)
        x   = torch.from_numpy(phi)
        with torch.no_grad():
            scores = self.net(x).squeeze(-1)
            probs_scoring = torch.softmax(scores, dim=0)

            if self.noise_walk:
                pw = float(0.5 * torch.sigmoid(self.noise_param).item())
                if random.random() < pw:
                    idx        = random.randint(0, k - 1)
                    by_policy  = False
                else:
                    idx        = int(torch.multinomial(probs_scoring, 1).item())
                    by_policy  = True
                prob_mix   = pw / k + (1.0 - pw) * probs_scoring[idx].item()
                log_prob   = math.log(max(prob_mix, 1e-9))
            else:
                idx        = int(torch.multinomial(probs_scoring, 1).item())
                log_prob   = float(torch.log(probs_scoring[idx]).item())
                by_policy  = True

        return candidates[idx], log_prob, by_policy

    def log_prob_phi(self, phi: np.ndarray) -> torch.Tensor:
        """
        Log-probabilities for each candidate from a pre-extracted feature matrix.
        Accounts for noise-walk mixture when noise_walk=True.
        Used by REINFORCE trainer — keeps gradient graph.
        """
        scores = self.net(torch.from_numpy(phi).float()).squeeze(-1)
        if self.noise_walk:
            k             = scores.shape[0]
            probs_scoring = torch.softmax(scores, dim=0)
            pw            = 0.5 * torch.sigmoid(self.noise_param)
            probs_mixture = pw / k + (1.0 - pw) * probs_scoring
            return torch.log(probs_mixture + 1e-9)
        return torch.log_softmax(scores, dim=0)

    def score_tensor(self, candidates: list[int], state: SLSState) -> torch.Tensor:
        """Training: raw MLP scores with gradient (no noise mixture)."""
        phi = extract_batch(candidates, state, self.feature_set, normalize=self.normalize)
        return self.net(torch.from_numpy(phi)).squeeze(-1)

    def score_logits(self, candidates: list[int], state: SLSState) -> torch.Tensor:
        return self.score_tensor(candidates, state)

    def score_phi(self, phi: np.ndarray) -> torch.Tensor:
        """Raw MLP scores from pre-extracted features. For backward compat."""
        return self.net(torch.from_numpy(phi).float()).squeeze(-1)

    def is_learnable(self) -> bool:
        return True
