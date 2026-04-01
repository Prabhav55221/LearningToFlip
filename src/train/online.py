"""
Online instance-specific adaptation (Mode 2).

Starting from offline-trained weights, interleave gradient steps with flip
decisions on a single test instance. A gradient step fires every delta flips
using the most recent k transitions as the mini-batch.

Key constraints:
  - Same flip budget as the frozen model (wall-clock reported separately)
  - Small learning rate to prevent catastrophic drift from offline init
  - delta = k by default (update when the k-step buffer is naturally full)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from src.sat.parser import CNFFormula
from src.sls.solver import SolveResult, StepRecord
from src.train.reinforce import KStepBuffer, REINFORCEConfig


@dataclass
class OnlineConfig:
    k: int = 10
    gamma: float = 0.5
    delta: int = 10         # gradient step every delta flips (default: delta = k)
    lr: float = 1e-5        # small LR to prevent forgetting


class OnlineAdapter:
    """
    Wraps a pre-trained policy and fine-tunes it during a single solve attempt.

    The solve loop is rewritten here (rather than reusing sls.solver.run_try)
    because gradient steps are interleaved and require access to the live
    trajectory buffer before the episode ends.
    """

    def __init__(self, policy: nn.Module, config: OnlineConfig) -> None:
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)
        self.buffer = KStepBuffer(config.k, config.gamma)
        self._baseline: float = 0.0

    def solve(self, formula: CNFFormula, max_flips: int) -> SolveResult:
        """
        Run a single SLS try with interleaved gradient updates.
        Returns a SolveResult (same interface as sls.solver).
        """
        raise NotImplementedError
