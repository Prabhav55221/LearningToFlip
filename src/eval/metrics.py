"""
Evaluation metrics for SLS experiments.

Primary metric: median flips-to-solution across test instances.
Secondary: solve rate %, CDF curves (run-length distributions).
Unsolved instances are treated as max_flips for median computation (standard SLS convention).
"""

import numpy as np
from dataclasses import dataclass
from src.sls.solver import SolveResult


@dataclass
class EvalSummary:
    median_flips: float
    mean_flips: float
    solve_rate: float       # fraction of instances solved within budget
    flip_counts: np.ndarray # raw per-instance flip counts (for CDF)


def summarise(results: list[SolveResult], max_flips: int) -> EvalSummary:
    """
    Aggregate a list of SolveResults into summary statistics.
    Unsolved instances contribute max_flips to the flip count distribution.
    """
    counts = np.array([r.n_flips if r.solved else max_flips for r in results])
    return EvalSummary(
        median_flips=float(np.median(counts)),
        mean_flips=float(np.mean(counts)),
        solve_rate=float(np.mean([r.solved for r in results])),
        flip_counts=counts,
    )


def cdf(flip_counts: np.ndarray, max_flips: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the empirical CDF of flips-to-solution (run-length distribution).
    Returns (x, y) arrays suitable for plotting.
    """
    x = np.sort(flip_counts)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y
