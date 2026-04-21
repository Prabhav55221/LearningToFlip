"""
REINFORCE with rolling k-step returns and EMA baseline.

Key design:
  - Per-step gradient updates (not end-of-episode)
  - Rolling deque of k (phi, idx, reward) entries; fires a gradient update when full
  - Buffer stores raw numpy features + chosen index — no live computation graphs
  - At update time, log_prob is recomputed from phi with CURRENT weights (avoids
    stale-graph RuntimeError from in-place optimizer weight modifications)
  - EMA baseline: subtract OLD value before gradient step, then update after
  - Generic warm-up via cross-entropy against min-break (works for linear and MLP)
  - Reward: make_count - break_count (per-step, not binary like Interian)

k-step return for flip at step t (fired when step t+k is observed):
    G_t = Σ_{i=0}^{k-1} γ^i · r_{t+i}
"""

import random
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.sat.parser import CNFFormula, parse_dimacs
from src.sat.state import SLSState
from src.policy.features import extract_batch

log = logging.getLogger(__name__)


@dataclass
class REINFORCEConfig:
    k: int = 10
    gamma: float = 0.5
    baseline_momentum: float = 0.99
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 60
    warmup_epochs: int = 5
    max_flips: int = 10_000
    val_every: int = 5
    max_tries_val: int = 10


class KStepBuffer:
    """
    Rolling deque of k (phi, idx, reward) tuples — no live tensors.

    Stores the pre-extracted feature matrix (numpy), the chosen action index,
    and the scalar reward. When the buffer fires, the caller recomputes log_prob
    from phi with CURRENT weights, avoiding stale-graph errors from in-place
    optimizer updates between the forward pass and the backward pass.

    push() returns (phi_t, idx_t, G_t) when the oldest entry has k rewards;
    returns None while filling up. Call reset() between episodes.
    """

    def __init__(self, k: int, gamma: float) -> None:
        self.k = k
        self.gamma = gamma
        self._buf: deque = deque(maxlen=k)

    def push(self, phi: np.ndarray, idx: int, reward: float):
        if len(self._buf) == self.k:
            phi_t, idx_t, _ = self._buf[0]
            G_t = float(sum(self.gamma ** i * self._buf[i][2] for i in range(self.k)))
            self._buf.append((phi, idx, reward))
            return phi_t, idx_t, G_t
        self._buf.append((phi, idx, reward))
        return None

    def reset(self) -> None:
        self._buf.clear()


class REINFORCETrainer:
    """
    Manages per-step REINFORCE updates with EMA baseline.
    Call reset() at the start of each new episode.
    """

    def __init__(self, policy: nn.Module, config: REINFORCEConfig) -> None:
        self.policy = policy
        self.config = config
        self.optimizer = torch.optim.AdamW(
            policy.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.buffer = KStepBuffer(config.k, config.gamma)
        self._baseline: float = 0.0

    def reset(self) -> None:
        self.buffer.reset()

    def step(self, phi: np.ndarray, idx: int, reward: float) -> dict | None:
        """
        Push (phi, idx, reward). If the buffer fires:
          - Recompute log_prob[idx] from stored phi with CURRENT weights
          - Compute k-step advantage and backprop
          - Update EMA baseline AFTER gradient step
        Returns a metrics dict or None if no update happened.
        """
        result = self.buffer.push(phi, idx, reward)
        if result is None:
            return None

        phi_t, idx_t, G_t = result

        # Recompute with current weights — no stale graph
        scores = self.policy.score_phi(phi_t)
        log_probs = torch.log_softmax(scores, dim=0)
        lp_t = log_probs[idx_t]

        advantage = G_t - self._baseline   # subtract OLD baseline
        loss = -(lp_t * advantage)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_baseline(G_t)         # update AFTER gradient step

        return {"loss": loss.item(), "return": G_t, "advantage": advantage}

    def _update_baseline(self, G_t: float) -> None:
        m = self.config.baseline_momentum
        self._baseline = m * self._baseline + (1.0 - m) * G_t


def _run_warmup_episode(
    formula: CNFFormula,
    policy: nn.Module,
    max_flips: int,
) -> torch.Tensor | None:
    """
    Cross-entropy warm-up: run with min-break as the oracle, push policy scores
    toward min-break selection. Works for any policy with a score_logits() method.
    Returns mean cross-entropy loss or None if the instance was already satisfied.
    """
    state = SLSState.random_init(formula)
    step_losses: list[torch.Tensor] = []

    for _ in range(max_flips):
        if state.is_solved:
            break

        candidates = state.random_unsat_clause()
        breaks = [state.break_count(v) for v in candidates]
        min_b = min(breaks)

        scores = policy.score_logits(candidates, state)
        log_probs = torch.log_softmax(scores, dim=0)

        n_min = sum(1 for b in breaks if b == min_b)
        target = torch.tensor(
            [1.0 / n_min if b == min_b else 0.0 for b in breaks],
            dtype=torch.float32,
        )
        step_losses.append(-torch.sum(target * log_probs))

        min_vars = [v for v, b in zip(candidates, breaks) if b == min_b]
        state.flip(random.choice(min_vars), by_policy=True)

    if not step_losses:
        return None
    return torch.stack(step_losses).mean()


def validate(
    val_formulas: list[CNFFormula],
    policy: nn.Module,
    config: REINFORCEConfig,
) -> float:
    """Returns median flips over the validation set (max_flips if unsolved)."""
    from src.sls.solver import solve
    flip_counts = []
    for formula in val_formulas:
        with torch.no_grad():
            result = solve(formula, policy, config.max_flips, config.max_tries_val)
        flip_counts.append(result.n_flips)
    return float(np.median(flip_counts))


def train(
    train_paths: list[Path],
    val_paths: list[Path],
    policy: nn.Module,
    config: REINFORCEConfig,
    save_dir: Path | None = None,
    run_name: str = "ours",
) -> nn.Module:
    """
    Train policy using REINFORCE with rolling k-step buffer.

    Phase 1 (warm-up): cross-entropy against min-break oracle.
    Phase 2 (REINFORCE): per-step gradient updates as the k-step buffer fills.

    Checkpoints by validation median flips; warm-up model is the initial best.
    Returns the best policy by validation.
    """
    log.info("Loading %d train + %d val formulas", len(train_paths), len(val_paths))
    train_formulas = [parse_dimacs(p) for p in train_paths]
    val_formulas = [parse_dimacs(p) for p in val_paths]

    # ------------------------------------------------------------------ #
    # Warm-up: cross-entropy against min-break oracle                     #
    # ------------------------------------------------------------------ #
    if config.warmup_epochs > 0:
        log.info("==> Warm-up phase (%d epochs)", config.warmup_epochs)
        warmup_opt = torch.optim.AdamW(
            policy.parameters(),
            lr=config.lr / 3,
            weight_decay=config.weight_decay,
        )
        for epoch in range(config.warmup_epochs):
            random.shuffle(train_formulas)
            losses: list[float] = []
            for formula in train_formulas:
                warmup_opt.zero_grad()
                loss = _run_warmup_episode(formula, policy, config.max_flips)
                if loss is not None:
                    loss.backward()
                    warmup_opt.step()
                    losses.append(loss.item())
            log.info(
                "  Warmup %d/%d  mean_loss=%.4f",
                epoch + 1, config.warmup_epochs,
                float(np.mean(losses)) if losses else 0.0,
            )

    # ------------------------------------------------------------------ #
    # Initial validation — warm-up model is the checkpoint to beat        #
    # ------------------------------------------------------------------ #
    val_med = validate(val_formulas, policy, config)
    log.info("  Pre-training val median: %.0f flips", val_med)
    best_val_median = val_med
    best_state_dict: dict = {k: v.clone() for k, v in policy.state_dict().items()}
    ckpt_path: Path | None = None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"best_{run_name}.pt"
        torch.save(best_state_dict, ckpt_path)
        log.info("  Saved warm-up model as initial best (val_median=%.0f)", best_val_median)

    # ------------------------------------------------------------------ #
    # REINFORCE phase: per-step rolling k-step buffer updates             #
    # ------------------------------------------------------------------ #
    log.info("==> REINFORCE phase (%d epochs)", config.epochs)
    trainer = REINFORCETrainer(policy, config)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        trainer.optimizer,
        max_lr=config.lr,
        steps_per_epoch=1,
        epochs=config.epochs,
        div_factor=5,
        final_div_factor=10,
    )

    for epoch in range(config.epochs):
        random.shuffle(train_formulas)
        all_losses: list[float] = []

        for formula in train_formulas:
            state = SLSState.random_init(formula)
            trainer.reset()

            for _ in range(config.max_flips):
                if state.is_solved:
                    break

                candidates = state.random_unsat_clause()
                phi = extract_batch(candidates, state, policy.feature_set)

                with torch.no_grad():
                    scores = policy.score_phi(phi)
                    idx = int(torch.multinomial(torch.softmax(scores, dim=0), 1).item())

                make_c, break_c = state.flip(candidates[idx], by_policy=True)
                metrics = trainer.step(phi, idx, float(make_c - break_c))

                if metrics is not None:
                    all_losses.append(metrics["loss"])

        scheduler.step()

        log.info(
            "Epoch %d/%d  mean_loss=%.4f",
            epoch + 1, config.epochs,
            float(np.mean(all_losses)) if all_losses else 0.0,
        )

        if (epoch + 1) % config.val_every == 0:
            val_med = validate(val_formulas, policy, config)
            log.info("  Val median: %.0f flips", val_med)

            if val_med < best_val_median:
                best_val_median = val_med
                best_state_dict = {k: v.clone() for k, v in policy.state_dict().items()}
                if ckpt_path is not None:
                    torch.save(best_state_dict, ckpt_path)
                    log.info("  Saved best model (val_median=%.0f)", best_val_median)

    policy.load_state_dict(best_state_dict)
    log.info("Restored best model (val_median=%.0f)", best_val_median)

    return policy
