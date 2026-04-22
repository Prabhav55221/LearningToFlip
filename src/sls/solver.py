"""
Policy-agnostic SLS loop.

Runs one SLS try (up to max_flips) from a random initial assignment.
Each step: pick random unsatisfied clause → call policy.select() → flip.
Returns a SolveResult with the full trajectory if requested (for REINFORCE).

Logging (controlled by caller via logging.setup()):
  DEBUG -- per-try result (try index, flips, solved/timeout)
  INFO  -- nothing here; per-instance logging is done by the caller
"""

import logging
from dataclasses import dataclass, field
from src.sat.parser import CNFFormula
from src.sat.state import SLSState
from src.policy.base import Policy

log = logging.getLogger(__name__)


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
    n_tries: int = 1
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
        var, log_prob, by_policy = policy.select(candidates, state)
        make_c, break_c = state.flip(var, by_policy=by_policy)

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
    for t in range(max_tries):
        result = run_try(formula, policy, max_flips, record_trajectory)
        result.n_tries = t + 1
        status = f"solved in {result.n_flips} flips" if result.solved else f"timeout ({max_flips} flips)"
        log.debug("  try %d/%d: %s", t + 1, max_tries, status)
        if result.solved:
            return result
    return result
