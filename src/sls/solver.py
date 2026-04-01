"""
Policy-agnostic SLS loop.

Runs one try (up to max_flips) given a formula, policy, and initial state.
Returns a SolveResult with the trajectory needed for training.
"""

from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn.functional as F
from src.sat.parser import CNFFormula
from src.sat.state import SLSState
from src.policy.base import Policy


@dataclass
class StepRecord:
    """One transition in the SLS trajectory, stored for REINFORCE."""
    var: int
    log_prob: float
    reward: float           # make - break


@dataclass
class SolveResult:
    solved: bool
    n_flips: int
    trajectory: list[StepRecord] = field(default_factory=list)


def run_try(
    formula: CNFFormula,
    policy: Policy,
    max_flips: int,
    record_trajectory: bool = False,
) -> SolveResult:
    """
    Run one SLS try from a random initial assignment.

    If record_trajectory=True, stores (var, log_prob, reward) at every step
    for use by the offline REINFORCE trainer. Set False for classical baselines
    to avoid the overhead.
    """
    state = SLSState.random_init(formula)
    trajectory: list[StepRecord] = []

    for _ in range(max_flips):
        if state.is_solved:
            return SolveResult(solved=True, n_flips=state.step, trajectory=trajectory)

        clause = state.random_unsat_clause()
        scores = policy.score(clause, state)

        # Softmax sampling
        probs = torch.softmax(torch.tensor(scores), dim=0)
        idx = torch.multinomial(probs, num_samples=1).item()
        var = clause[idx]
        log_prob = torch.log(probs[idx]).item()

        make_count, break_count = state.flip(var)
        reward = make_count - break_count

        if record_trajectory:
            trajectory.append(StepRecord(var=var, log_prob=log_prob, reward=reward))

    return SolveResult(solved=False, n_flips=max_flips, trajectory=trajectory)


def solve(
    formula: CNFFormula,
    policy: Policy,
    max_flips: int,
    max_tries: int,
    record_trajectory: bool = False,
) -> SolveResult:
    """
    Run up to max_tries independent restarts. Returns the first solved result,
    or the last failed result if none succeed.
    """
    result = SolveResult(solved=False, n_flips=0)
    for _ in range(max_tries):
        result = run_try(formula, policy, max_flips, record_trajectory)
        if result.solved:
            return result
    return result
