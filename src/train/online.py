"""
Online instance-specific adaptation.

This module exposes exactly two online adaptation methods for generalization
evaluation:

  - OnlineKLAdapter: k-step REINFORCE with a KL anchor to the frozen offline
    policy on every update.
  - OnlineSuccessKLAdapter: update only from successful trajectories, then keep
    solving with the adapted weights on later tries.

Both adapters reset to the offline checkpoint between instances by default.
"""

import copy
import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.policy.features import extract_batch
from src.sat.parser import CNFFormula
from src.sat.state import SLSState
from src.sls.solver import SolveResult
from src.train.reinforce import KStepBuffer

log = logging.getLogger(__name__)


def _clone_reference_policy(policy: nn.Module) -> nn.Module:
    ref = copy.deepcopy(policy)
    ref.eval()
    return ref


def _kl_to_reference(
    current_log_probs: torch.Tensor,
    reference_policy: nn.Module,
    phi: np.ndarray,
) -> torch.Tensor:
    with torch.no_grad():
        ref_log_probs = reference_policy.log_prob_phi(phi)
    probs = torch.exp(current_log_probs)
    return torch.sum(probs * (current_log_probs - ref_log_probs))


@dataclass
class OnlineKLConfig:
    k: int = 10
    gamma: float = 0.5
    lr: float = 1e-5
    entropy_coef: float = 0.01
    kl_anchor_coef: float = 0.05
    normalize_reward: bool = False
    max_grad_norm: float = 1.0
    baseline_momentum: float = 0.99


@dataclass
class OnlineEvalResult:
    solved: bool
    best_flips: int
    cumulative_flips: int
    n_tries: int


class OnlineKLAdapter:
    """Per-step k-step REINFORCE with a KL anchor to the frozen offline policy."""

    def __init__(self, policy: nn.Module, config: OnlineKLConfig) -> None:
        self.policy = policy
        self.config = config
        self._offline_sd = {k: v.clone() for k, v in policy.state_dict().items()}
        self._reference_policy = _clone_reference_policy(policy)
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)
        self.buffer = KStepBuffer(config.k, config.gamma)
        self._baseline: float = 0.0

    def reset_to_offline(self) -> None:
        self.policy.load_state_dict(self._offline_sd)
        self._reference_policy.load_state_dict(self._offline_sd)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.config.lr)
        self.buffer.reset()
        self._baseline = 0.0

    def _update(self, phi: np.ndarray, idx: int, G_t: float) -> None:
        log_probs = self.policy.log_prob_phi(phi)
        lp = log_probs[idx]
        entropy = -torch.sum(torch.exp(log_probs) * log_probs)
        advantage = G_t - self._baseline
        loss = -(lp * advantage) - self.config.entropy_coef * entropy

        if self.config.kl_anchor_coef > 0.0:
            kl_term = _kl_to_reference(log_probs, self._reference_policy, phi)
            loss = loss + self.config.kl_anchor_coef * kl_term

        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        m = self.config.baseline_momentum
        self._baseline = m * self._baseline + (1.0 - m) * G_t

    def _run_try(self, formula: CNFFormula, max_flips: int) -> tuple[bool, int]:
        self.buffer.reset()
        state = SLSState.random_init(formula)

        for step in range(max_flips):
            if state.is_solved:
                return True, step

            candidates = state.random_unsat_clause()
            phi = extract_batch(
                candidates,
                state,
                self.policy.feature_set,
                normalize=self.policy.normalize,
            )

            with torch.no_grad():
                log_probs_inf = self.policy.log_prob_phi(phi)
                idx = int(torch.multinomial(torch.exp(log_probs_inf), 1).item())

            make_c, break_c = state.flip(candidates[idx], by_policy=True)
            reward = float(make_c - break_c)
            if self.config.normalize_reward:
                reward = math.tanh(reward)

            fired = self.buffer.push(phi, idx, reward)
            if fired is not None:
                self._update(fired[0], fired[1], fired[2])

        return False, max_flips

    def solve(
        self,
        formula: CNFFormula,
        max_flips: int,
        max_tries: int = 10,
        reset: bool = True,
    ) -> SolveResult:
        if reset:
            self.reset_to_offline()

        for t in range(max_tries):
            solved, n_flips = self._run_try(formula, max_flips)
            log.debug(
                "  try %d/%d: %s",
                t + 1,
                max_tries,
                f"solved in {n_flips}" if solved else f"timeout ({max_flips})",
            )
            if solved:
                return SolveResult(solved=True, n_flips=n_flips, n_tries=t + 1)

        return SolveResult(solved=False, n_flips=max_flips, n_tries=max_tries)

    def evaluate(
        self,
        formula: CNFFormula,
        max_flips: int,
        max_tries: int = 10,
        reset: bool = True,
    ) -> OnlineEvalResult:
        if reset:
            self.reset_to_offline()

        solved_any = False
        best_flips = max_flips
        cumulative_flips = 0

        for t in range(max_tries):
            solved, n_flips = self._run_try(formula, max_flips)
            cumulative_flips += n_flips
            log.debug(
                "  try %d/%d: %s",
                t + 1,
                max_tries,
                f"solved in {n_flips}" if solved else f"timeout ({max_flips})",
            )
            if solved and ((not solved_any) or (n_flips < best_flips)):
                solved_any = True
                best_flips = n_flips

        return OnlineEvalResult(
            solved=solved_any,
            best_flips=best_flips,
            cumulative_flips=cumulative_flips,
            n_tries=max_tries,
        )


@dataclass
class OnlineSuccessKLConfig:
    gamma: float = 0.5
    lr: float = 1e-5
    kl_anchor_coef: float = 0.02
    max_grad_norm: float = 1.0


class OnlineSuccessKLAdapter:
    """
    Update only from successful trajectories, then keep solving with the
    adapted weights on later tries.
    """

    def __init__(self, policy: nn.Module, config: OnlineSuccessKLConfig) -> None:
        self.policy = policy
        self.config = config
        self._offline_sd = {k: v.clone() for k, v in policy.state_dict().items()}
        self._reference_policy = _clone_reference_policy(policy)
        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr)

    def reset_to_offline(self) -> None:
        self.policy.load_state_dict(self._offline_sd)
        self._reference_policy.load_state_dict(self._offline_sd)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.config.lr)

    def _run_try(
        self,
        formula: CNFFormula,
        max_flips: int,
    ) -> tuple[bool, int, list[tuple[np.ndarray, int]]]:
        state = SLSState.random_init(formula)
        trajectory: list[tuple[np.ndarray, int]] = []

        for step in range(max_flips):
            if state.is_solved:
                return True, step, trajectory

            candidates = state.random_unsat_clause()
            phi = extract_batch(
                candidates,
                state,
                self.policy.feature_set,
                normalize=self.policy.normalize,
            )

            with torch.no_grad():
                log_probs_inf = self.policy.log_prob_phi(phi)
                idx = int(torch.multinomial(torch.exp(log_probs_inf), 1).item())

            state.flip(candidates[idx], by_policy=True)
            trajectory.append((phi, idx))

        return False, max_flips, trajectory

    def _fine_tune(self, trajectory: list[tuple[np.ndarray, int]]) -> float:
        T = len(trajectory)
        if T == 0:
            return 0.0

        step_losses: list[torch.Tensor] = []
        for t, (phi, idx) in enumerate(trajectory):
            G_t = self.config.gamma ** (T - 1 - t)
            log_probs = self.policy.log_prob_phi(phi)
            loss_t = -G_t * log_probs[idx]

            if self.config.kl_anchor_coef > 0.0:
                kl_term = _kl_to_reference(log_probs, self._reference_policy, phi)
                loss_t = loss_t + self.config.kl_anchor_coef * kl_term

            step_losses.append(loss_t)

        loss = torch.stack(step_losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def solve(
        self,
        formula: CNFFormula,
        max_flips: int,
        max_tries: int = 10,
        reset: bool = True,
    ) -> SolveResult:
        if reset:
            self.reset_to_offline()

        best_result = SolveResult(solved=False, n_flips=max_flips, n_tries=max_tries)
        best_try = max_tries

        for t in range(max_tries):
            solved, n_flips, trajectory = self._run_try(formula, max_flips)
            if solved:
                if (not best_result.solved) or (n_flips < best_result.n_flips):
                    best_result = SolveResult(solved=True, n_flips=n_flips, n_tries=t + 1)
                    best_try = t + 1
                loss = self._fine_tune(trajectory)
                log.debug(
                    "  try %d/%d: solved in %d  fine_tune_loss=%.4f",
                    t + 1,
                    max_tries,
                    n_flips,
                    loss,
                )
            else:
                log.debug("  try %d/%d: timeout (%d)", t + 1, max_tries, max_flips)

        if best_result.solved:
            best_result.n_tries = best_try
            return best_result
        return SolveResult(solved=False, n_flips=max_flips, n_tries=max_tries)

    def evaluate(
        self,
        formula: CNFFormula,
        max_flips: int,
        max_tries: int = 10,
        reset: bool = True,
    ) -> OnlineEvalResult:
        if reset:
            self.reset_to_offline()

        solved_any = False
        best_flips = max_flips
        cumulative_flips = 0

        for t in range(max_tries):
            solved, n_flips, trajectory = self._run_try(formula, max_flips)
            cumulative_flips += n_flips
            if solved:
                if (not solved_any) or (n_flips < best_flips):
                    solved_any = True
                    best_flips = n_flips
                loss = self._fine_tune(trajectory)
                log.debug(
                    "  try %d/%d: solved in %d  fine_tune_loss=%.4f",
                    t + 1,
                    max_tries,
                    n_flips,
                    loss,
                )
            else:
                log.debug("  try %d/%d: timeout (%d)", t + 1, max_tries, max_flips)

        return OnlineEvalResult(
            solved=solved_any,
            best_flips=best_flips,
            cumulative_flips=cumulative_flips,
            n_tries=max_tries,
        )
