"""
Online instance-specific adaptation.

Starting from offline-trained weights, interleave gradient steps with flip
decisions on a single test instance. A gradient update fires every time the
k-step buffer is full (same rolling-buffer REINFORCE as offline training).

Design choices vs offline training:
  - Small LR (default 1e-5) prevents catastrophic forgetting of offline init
  - Gradient clipping (max_grad_norm=1.0) tames the wild loss from sparse rewards
  - Entropy regularization kept (default 0.01) to maintain exploration
  - Policy weights reset to offline checkpoint between instances (default True)
    so each instance starts from the same initialisation; set reset=False
    for continual adaptation across instances

The solve() method returns a SolveResult with the same interface as
src.sls.solver.solve(), so eval_generalization.py can handle it uniformly.
"""

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.sat.parser import CNFFormula
from src.sat.state import SLSState
from src.sls.solver import SolveResult
from src.policy.features import extract_batch
from src.train.reinforce import KStepBuffer

log = logging.getLogger(__name__)


@dataclass
class OnlineConfig:
    k: int   = 10
    gamma: float = 0.5
    lr: float    = 1e-5
    entropy_coef: float  = 0.01
    normalize_reward: bool = False
    max_grad_norm: float = 1.0
    baseline_momentum: float = 0.99


class OnlineAdapter:
    """
    Wraps a pre-trained policy and fine-tunes it during solving.

    Usage:
        adapter = OnlineAdapter(policy, config)
        result  = adapter.solve(formula, max_flips, max_tries)

    The adapter owns the optimizer; call reset_to_offline() between
    instances if you want independent per-instance adaptation.
    """

    def __init__(self, policy: nn.Module, config: OnlineConfig) -> None:
        self.policy  = policy
        self.config  = config
        # Snapshot offline weights so we can reset between instances
        self._offline_sd = {k: v.clone() for k, v in policy.state_dict().items()}
        self.optimizer    = torch.optim.AdamW(policy.parameters(), lr=config.lr)
        self.buffer       = KStepBuffer(config.k, config.gamma)
        self._baseline: float = 0.0

    # ── Weight management ────────────────────────────────────────────────────

    def reset_to_offline(self) -> None:
        """Restore offline weights and fresh optimizer state."""
        self.policy.load_state_dict(self._offline_sd)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.config.lr)
        self._baseline = 0.0

    # ── Gradient update ──────────────────────────────────────────────────────

    def _update(self, phi: np.ndarray, idx: int, G_t: float) -> float:
        """
        One REINFORCE step with entropy reg and gradient clipping.
        Recomputes log_prob from stored phi with CURRENT weights (no stale graph).
        Returns scalar loss.
        """
        log_prob = self.policy.log_prob_phi(phi)
        lp       = log_prob[idx]
        entropy  = -torch.sum(torch.exp(log_prob) * log_prob)

        advantage = G_t - self._baseline
        loss      = -(lp * advantage) - self.config.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        # EMA baseline — update AFTER gradient step
        m = self.config.baseline_momentum
        self._baseline = m * self._baseline + (1.0 - m) * G_t

        return loss.item()

    # ── Single try ───────────────────────────────────────────────────────────

    def _run_try(self, formula: CNFFormula, max_flips: int) -> tuple[bool, int]:
        """
        One random-restart try with interleaved gradient updates.
        Returns (solved, n_flips).
        """
        self.buffer.reset()
        state = SLSState.random_init(formula)

        for step in range(max_flips):
            if state.is_solved:
                return True, step

            candidates   = state.random_unsat_clause()
            phi          = extract_batch(candidates, state, self.policy.feature_set, normalize=self.policy.normalize)

            # Sample action via mixture (respects noise_walk) — recomputed at update time
            with torch.no_grad():
                log_probs_inf = self.policy.log_prob_phi(phi)
                idx           = int(torch.multinomial(torch.exp(log_probs_inf), 1).item())

            make_c, break_c = state.flip(candidates[idx], by_policy=True)
            reward          = float(make_c - break_c)
            if self.config.normalize_reward:
                reward = math.tanh(reward)

            fired = self.buffer.push(phi, idx, reward)
            if fired is not None:
                phi_t, idx_t, G_t = fired
                self._update(phi_t, idx_t, G_t)

        return False, max_flips

    # ── Public solve interface ───────────────────────────────────────────────

    def solve(
        self,
        formula:   CNFFormula,
        max_flips: int,
        max_tries: int = 10,
        reset:     bool = True,
    ) -> SolveResult:
        """
        Solve with online adaptation. Matches the interface of sls.solver.solve().

        reset=True  — restore offline weights before this instance (default)
        reset=False — carry adapted weights over from previous instance
        """
        if reset:
            self.reset_to_offline()

        for t in range(max_tries):
            solved, n_flips = self._run_try(formula, max_flips)
            log.debug("  try %d/%d: %s", t + 1, max_tries,
                      f"solved in {n_flips}" if solved else f"timeout ({max_flips})")
            if solved:
                return SolveResult(solved=True, n_flips=n_flips, n_tries=t + 1)

        return SolveResult(solved=False, n_flips=max_flips, n_tries=max_tries)
