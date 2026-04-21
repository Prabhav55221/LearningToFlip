"""
Linear scoring policy — exact replication of Interian & Bernardini (KR 2023).

Scoring function:  f_θ(x) = θ_0 + Σ θ_i · φ_i(x, s)
Noise:             p_w = 0.5 · sigmoid(w_0)  (learned scalar, jointly with θ)

pickVar logic (Algorithm 3 in the paper):
  - With probability p_w: pick a uniformly random variable (noise walk) → by_policy=False
  - Otherwise: sample from softmax(f_θ) over the clause candidates  → by_policy=True

The by_policy flag tells the solver whether to update age2 (policy recency)
in the state, which is used by the delta2/policy_last5/policy_last10 features.

For REINFORCE training, use score_logprobs() which returns the full mixture
log-probabilities for all candidates WITH gradient tracking.
"""

import random
import math
import numpy as np
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
        # Noise parameter w: p_w = 0.5 * sigmoid(w)
        # Initialized to 0 → p_w = 0.25 at start
        self.noise_param = nn.Parameter(torch.zeros(1))

    @property
    def noise_prob(self) -> float:
        """Current learned noise probability p_w ∈ (0, 0.5)."""
        return float(0.5 * torch.sigmoid(self.noise_param).item())

    def select(self, candidates: list[int], state: SLSState) -> tuple[int, float, bool]:
        """
        Inference: decide noise vs. scoring branch, sample action, return 3-tuple.
        Uses torch.no_grad() — not suitable for gradient computation.
        """
        k = len(candidates)
        phi = extract_batch(candidates, state, self.feature_set)
        x = torch.from_numpy(phi).float()

        with torch.no_grad():
            scores = self.linear(x).squeeze(-1)            # (k,)
            probs_scoring = torch.softmax(scores, dim=0).numpy()  # (k,)
            pw = float(0.5 * torch.sigmoid(self.noise_param).item())

        # Explicit branch decision (needed to set by_policy correctly)
        if random.random() < pw:
            idx = random.randint(0, k - 1)
            by_policy = False
        else:
            idx = int(np.random.choice(k, p=probs_scoring / probs_scoring.sum()))
            by_policy = True

        # Log probability under the MIXTURE distribution (for trajectory recording)
        prob_mix = pw / k + (1.0 - pw) * probs_scoring[idx]
        log_prob = math.log(max(prob_mix, 1e-9))

        return candidates[idx], log_prob, by_policy

    def score_logprobs(self, candidates: list[int], state: SLSState) -> torch.Tensor:
        """
        Training: return mixture log-probabilities for all candidates WITH gradients.
        Used by the REINFORCE episode runner to collect differentiable log_prob tensors.

        p_mixture(x) = p_w / k  +  (1 - p_w) · softmax(f_θ(x))
        """
        k = len(candidates)
        phi = extract_batch(candidates, state, self.feature_set)
        x = torch.from_numpy(phi).float()
        scores = self.linear(x).squeeze(-1)              # (k,)
        probs_scoring = torch.softmax(scores, dim=0)     # (k,)
        pw = 0.5 * torch.sigmoid(self.noise_param)       # scalar tensor
        probs_mixture = pw / k + (1.0 - pw) * probs_scoring  # (k,)
        return torch.log(probs_mixture + 1e-9)           # (k,)  — log probs with grad

    def score_logits(self, candidates: list[int], state: SLSState) -> torch.Tensor:
        """Training: raw linear scores WITH gradient. No noise/mixture. Used by our REINFORCE."""
        phi = extract_batch(candidates, state, self.feature_set)
        x = torch.from_numpy(phi).float()
        return self.linear(x).squeeze(-1)

    def score_phi(self, phi: np.ndarray) -> torch.Tensor:
        """Score from a pre-extracted feature matrix (k, n_features). Used by REINFORCE trainer."""
        return self.linear(torch.from_numpy(phi).float()).squeeze(-1)

    def is_learnable(self) -> bool:
        return True
