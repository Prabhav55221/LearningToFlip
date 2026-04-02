"""
Linear scoring policy — exact replication of Interian & Bernardini (KR 2023).

score(x) = w · φ(x, state)

Trained via REINFORCE. The linear model is a strict special case of the MLP
(zero hidden layers) and serves as the primary ablation baseline.
"""

import torch
import torch.nn as nn
from src.sat.state import SLSState
from src.policy.features import extract_batch, FEATURE_SETS


class LinearPolicy(nn.Module):
    def __init__(self, feature_set: str = "interian") -> None:
        super().__init__()
        n_features = len(FEATURE_SETS[feature_set])
        self.feature_set = feature_set
        self.linear = nn.Linear(n_features, 1, bias=True)

    def select(self, candidates: list[int], state: SLSState) -> tuple[int, float]:
        """Inference: softmax sample, no gradient."""
        phi = extract_batch(candidates, state, self.feature_set)
        x = torch.from_numpy(phi)
        with torch.no_grad():
            scores = self.linear(x).squeeze(-1)
        probs = torch.softmax(scores, dim=0)
        idx = int(torch.multinomial(probs, num_samples=1).item())
        log_prob = float(torch.log(probs[idx]).item())
        return candidates[idx], log_prob

    def score_tensor(self, candidates: list[int], state: SLSState) -> torch.Tensor:
        """Training: scores with gradient tracking for REINFORCE."""
        phi = extract_batch(candidates, state, self.feature_set)
        x = torch.from_numpy(phi)
        return self.linear(x).squeeze(-1)  # shape (n_candidates,)

    def is_learnable(self) -> bool:
        return True
