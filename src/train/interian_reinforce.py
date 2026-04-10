"""
Faithful REINFORCE implementation for Interian & Bernardini (KR 2023).

Differences from our method (src/train/reinforce.py):
  - Reward:   binary 1/0 (solved or not), NOT make−break
  - Returns:  full-episode γ^(T−t) · r, NOT k-step
  - Gradient: only from solved episodes (r=0 → zero gradient)
  - Noise:    learned p_w = 0.5·sigmoid(w), jointly trained with scoring weights
  - Warm-up:  5 epochs of cross-entropy to mimic WalkSAT min-break selection

Training procedure (Section 4.3):
  - AdamW optimizer (separate LR for scoring weights and noise parameter)
  - OneCycleLR scheduler, one step per epoch
  - 60 REINFORCE epochs after 5 warm-up epochs
  - Per-instance gradient updates within each epoch
  - Validation every val_every epochs; save best checkpoint by median flips
"""

import random
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.sat.parser import CNFFormula, parse_dimacs
from src.sat.state import SLSState
from src.policy.linear import LinearPolicy
from src.policy.features import extract_batch

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class InterianConfig:
    # Training schedule
    epochs: int = 60
    warmup_epochs: int = 5
    max_lr: float = 1e-3          # peak LR for scoring weights (OneCycleLR)
    noise_lr: float = 1e-3        # LR for noise parameter (constant)
    weight_decay: float = 1e-5
    # RL
    gamma: float = 0.5            # discount factor (Interian: stable across 0.1–0.9)
    max_flips: int = 10_000       # per try during training (one try per instance)
    # Validation
    val_every: int = 5
    max_tries_val: int = 10       # restarts during validation


# --------------------------------------------------------------------------- #
# Episode runners                                                              #
# --------------------------------------------------------------------------- #

def run_warmup_episode(
    formula: CNFFormula,
    policy: LinearPolicy,
    max_flips: int,
) -> torch.Tensor | None:
    """
    Run a WalkSAT (min-break) episode.  At each step, compute cross-entropy
    loss pushing the policy's scoring weights toward min-break selection.
    The episode progresses using min-break (not the policy) so it is likely
    to solve the instance and provide useful training steps.

    Returns mean cross-entropy loss over all steps, or None if no steps taken.
    """
    state = SLSState.random_init(formula)
    step_losses: list[torch.Tensor] = []

    for _ in range(max_flips):
        if state.is_solved:
            break

        candidates = state.random_unsat_clause()
        breaks = [state.break_count(v) for v in candidates]
        min_b = min(breaks)

        # Policy log-probs over candidates (with gradient, scoring only — no noise)
        phi = extract_batch(candidates, state, policy.feature_set)
        x = torch.from_numpy(phi).float()
        scores = policy.linear(x).squeeze(-1)          # (k,)
        log_probs = torch.log_softmax(scores, dim=0)   # (k,)

        # Soft target: uniform distribution over min-break candidates
        n_min = sum(1 for b in breaks if b == min_b)
        target = torch.tensor(
            [1.0 / n_min if b == min_b else 0.0 for b in breaks],
            dtype=torch.float32,
        )
        step_losses.append(-torch.sum(target * log_probs))

        # Flip the min-break variable to progress the episode
        min_vars = [v for v, b in zip(candidates, breaks) if b == min_b]
        state.flip(random.choice(min_vars), by_policy=True)

    if not step_losses:
        return None
    return torch.stack(step_losses).mean()


def run_reinforce_episode(
    formula: CNFFormula,
    policy: LinearPolicy,
    config: InterianConfig,
) -> torch.Tensor | None:
    """
    Run one REINFORCE episode (one random-restart try).

    Collects mixture log-prob tensors WITH gradients at each step, then
    computes the full-episode policy gradient loss if the episode is solved.

    Returns loss tensor if solved, None if not solved (r=0 → no gradient).
    """
    state = SLSState.random_init(formula)
    log_probs: list[torch.Tensor] = []   # scalar tensors, one per flip

    for _ in range(config.max_flips):
        if state.is_solved:
            break

        candidates = state.random_unsat_clause()
        k = len(candidates)

        # Mixture log-probs WITH gradient (needed for REINFORCE update)
        all_log_probs = policy.score_logprobs(candidates, state)   # (k,)

        # Sample action WITHOUT gradient (action selection is not differentiated)
        with torch.no_grad():
            pw = float(0.5 * torch.sigmoid(policy.noise_param).item())
            if random.random() < pw:
                idx = random.randint(0, k - 1)
                by_policy = False
            else:
                probs_scoring = torch.softmax(
                    policy.linear(torch.from_numpy(
                        extract_batch(candidates, state, policy.feature_set)
                    ).float()).squeeze(-1),
                    dim=0,
                )
                idx = int(torch.multinomial(probs_scoring, num_samples=1).item())
                by_policy = True

        log_probs.append(all_log_probs[idx])   # scalar tensor WITH grad
        state.flip(candidates[idx], by_policy=by_policy)

    if not state.is_solved:
        return None   # r = 0: no gradient contribution

    T = len(log_probs)
    if T == 0:
        return None

    # Full-episode discounted returns: G_t = γ^(T−t)  (r=1 since solved)
    discounts = torch.tensor(
        [config.gamma ** (T - t) for t in range(T)], dtype=torch.float32
    )
    loss = -(discounts * torch.stack(log_probs)).mean()
    return loss


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #

def validate(
    val_formulas: list[CNFFormula],
    policy: LinearPolicy,
    config: InterianConfig,
) -> float:
    """Returns median flips over the validation set (max_flips if unsolved)."""
    from src.sls.solver import solve
    flip_counts = []
    for formula in val_formulas:
        with torch.no_grad():
            result = solve(formula, policy, config.max_flips, config.max_tries_val)
        flip_counts.append(result.n_flips)
    return float(np.median(flip_counts))


# --------------------------------------------------------------------------- #
# Training loop                                                                #
# --------------------------------------------------------------------------- #

def train(
    train_paths: list[Path],
    val_paths: list[Path],
    config: InterianConfig,
    save_dir: Path | None = None,
) -> LinearPolicy:
    """
    Train a LinearPolicy using Interian's REINFORCE procedure.

    Args:
        train_paths:  paths to training .cnf files (1900 for full dataset)
        val_paths:    paths to validation .cnf files (100 for full dataset)
        config:       hyperparameters
        save_dir:     directory to save best model checkpoint

    Returns the best model by validation median flips.
    """
    log.info("Loading %d train + %d val formulas", len(train_paths), len(val_paths))
    train_formulas = [parse_dimacs(p) for p in train_paths]
    val_formulas = [parse_dimacs(p) for p in val_paths]

    policy = LinearPolicy(feature_set="interian")

    # ------------------------------------------------------------------ #
    # Warm-up: cross-entropy loss to mimic WalkSAT min-break (Section 3.5)
    # ------------------------------------------------------------------ #
    log.info("==> Warm-up phase (%d epochs)", config.warmup_epochs)
    warmup_opt = torch.optim.AdamW(
        policy.linear.parameters(),
        lr=config.max_lr / 3,
        weight_decay=config.weight_decay,
    )
    for epoch in range(config.warmup_epochs):
        random.shuffle(train_formulas)
        losses: list[float] = []
        for formula in train_formulas:
            warmup_opt.zero_grad()
            loss = run_warmup_episode(formula, policy, config.max_flips)
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
    # REINFORCE phase                                                      #
    # ------------------------------------------------------------------ #
    log.info("==> REINFORCE phase (%d epochs)", config.epochs)

    # Separate parameter groups to allow independent LR scheduling
    optimizer = torch.optim.AdamW(
        [
            {"params": list(policy.linear.parameters()), "lr": config.max_lr / 3},
            {"params": [policy.noise_param], "lr": config.noise_lr},
        ],
        weight_decay=config.weight_decay,
    )
    # One step per epoch: LR peaks at epoch ~30, decays by epoch 60
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config.max_lr, config.noise_lr],
        steps_per_epoch=1,
        epochs=config.epochs,
        div_factor=5,
        final_div_factor=10,
    )

    best_val_median = float("inf")
    best_state_dict: dict | None = None

    # Initial validation before training starts
    val_med = validate(val_formulas, policy, config)
    log.info("  Pre-training val median: %.0f flips", val_med)

    for epoch in range(config.epochs):
        random.shuffle(train_formulas)
        solved_count = 0
        rl_losses: list[float] = []

        for formula in train_formulas:
            optimizer.zero_grad()
            loss = run_reinforce_episode(formula, policy, config)
            if loss is not None:
                loss.backward()
                optimizer.step()
                solved_count += 1
                rl_losses.append(loss.item())

        scheduler.step()   # one scheduler step per epoch (OneCycleLR)

        pw = policy.noise_prob
        log.info(
            "Epoch %d/%d  solved=%d/%d  mean_loss=%.4f  pw=%.4f",
            epoch + 1, config.epochs,
            solved_count, len(train_formulas),
            float(np.mean(rl_losses)) if rl_losses else 0.0,
            pw,
        )

        # Validation
        if (epoch + 1) % config.val_every == 0:
            val_med = validate(val_formulas, policy, config)
            log.info("  Val median: %.0f flips", val_med)

            if val_med < best_val_median:
                best_val_median = val_med
                best_state_dict = {k: v.clone() for k, v in policy.state_dict().items()}
                if save_dir is not None:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(best_state_dict, save_dir / "best_interian.pt")
                    log.info("  Saved best model (val_median=%.0f)", best_val_median)

    # Restore best checkpoint
    if best_state_dict is not None:
        policy.load_state_dict(best_state_dict)
        log.info("Restored best model (val_median=%.0f)", best_val_median)

    return policy
