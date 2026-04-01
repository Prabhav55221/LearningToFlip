"""
REINFORCE with k-step discounted returns and mean baseline.

Offline training loop:
  - Sample a batch of instances from the training set
  - For each instance, run one SLS try with trajectory recording
  - Compute k-step returns from the buffered trajectory
  - Update policy parameters via policy gradient

k-step return for flip at step t:
    G_t = sum_{i=0}^{k-1} gamma^i * r_{t+i}

Update fires at step t+k once r_t ... r_{t+k-1} are observed.
Baseline b is an exponential moving average of recent G_t values.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from src.sls.solver import StepRecord


@dataclass
class REINFORCEConfig:
    k: int = 10                 # return horizon
    gamma: float = 0.5          # discount factor (matches Interian)
    baseline_momentum: float = 0.99
    lr: float = 1e-3


class KStepBuffer:
    """
    Rolling buffer that computes k-step returns from a recorded trajectory.
    Call compute() after a full episode to get (log_probs, advantages) tensors.
    """

    def __init__(self, k: int, gamma: float) -> None:
        self.k = k
        self.gamma = gamma

    def compute(self, trajectory: list[StepRecord]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            log_probs: shape (T-k,)
            returns:   shape (T-k,)  — k-step discounted returns
        """
        raise NotImplementedError


class REINFORCETrainer:
    def __init__(self, policy: nn.Module, config: REINFORCEConfig) -> None:
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)
        self.buffer = KStepBuffer(config.k, config.gamma)
        self._baseline: float = 0.0

    def update(self, trajectory: list[StepRecord]) -> dict:
        """
        Compute k-step returns, subtract baseline, compute policy gradient loss,
        step optimizer. Returns a dict of training metrics (loss, mean return, etc.).
        """
        raise NotImplementedError

    def _update_baseline(self, mean_return: float) -> None:
        m = self.config.baseline_momentum
        self._baseline = m * self._baseline + (1 - m) * mean_return
