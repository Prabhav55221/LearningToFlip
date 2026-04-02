"""
Policy-agnostic SLS loop.

Runs one SLS try (up to max_flips) from a random initial assignment.
Each step: pick random unsatisfied clause → call policy.select() → flip.
Returns a SolveResult with the full trajectory if requested (for REINFORCE).
"""

from dataclasses import dataclass, field
from src.sat.parser import CNFFormula
from src.sat.state import SLSState
from src.policy.base import Policy


@dataclass
class StepRecord:
    """One transition recorded during an SLS trajectory, used for REINFORCE."""
    var: int
    log_prob: float   # 0.0 for classical baselines; log π(a|s) for learned policies
    reward: float     # make_count - break_count


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
    One SLS try from a fresh random assignment.
    Set record_trajectory=False for classical baselines to skip the overhead.
    """
    state = SLSState.random_init(formula)
    trajectory: list[StepRecord] = []

    for _ in range(max_flips):
        if state.is_solved:
            return SolveResult(solved=True, n_flips=state.step, trajectory=trajectory)

        candidates = state.random_unsat_clause()
        var, log_prob = policy.select(candidates, state)
        make_c, break_c = state.flip(var)

        if record_trajectory:
            trajectory.append(StepRecord(
                var=var,
                log_prob=log_prob,
                reward=float(make_c - break_c),
            ))

    return SolveResult(solved=False, n_flips=max_flips, trajectory=trajectory)


def solve(
    formula: CNFFormula,
    policy: Policy,
    max_flips: int,
    max_tries: int,
    record_trajectory: bool = False,
) -> SolveResult:
    """
    Run up to max_tries independent random-restart tries.
    Returns the first solved result, or the last failed result.
    """
    result = SolveResult(solved=False, n_flips=0)
    for _ in range(max_tries):
        result = run_try(formula, policy, max_flips, record_trajectory)
        if result.solved:
            return result
    return result
