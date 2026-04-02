"""
Classical SLS baselines. No learnable parameters.

Both implement the Policy protocol identically to learned policies and can be
dropped into the SLS loop and evaluation runner without any special casing.
"""

import random
from src.sat.state import SLSState


class MinBreak:
    """
    WalkSAT min-break: flip the candidate with the lowest break count.
    Ties broken uniformly at random.
    """

    def select(self, candidates: list[int], state: SLSState) -> tuple[int, float]:
        breaks = [state.break_count(v) for v in candidates]
        min_b = min(breaks)
        tied = [v for v, b in zip(candidates, breaks) if b == min_b]
        return random.choice(tied), 0.0

    def is_learnable(self) -> bool:
        return False


class NoveltyPlus:
    """
    Novelty+: with probability p pick a uniformly random candidate (random walk).
    Otherwise apply the Novelty rule:
      - Among candidates with minimum break count, exclude the most recently
        flipped variable unless it is the only min-break candidate.
      - Pick uniformly among the remaining min-break candidates.

    p=0.0 gives pure Novelty (no random walk).
    Default p=0.1 matches the standard Novelty+ setting.
    """

    def __init__(self, p: float = 0.1) -> None:
        self.p = p

    def select(self, candidates: list[int], state: SLSState) -> tuple[int, float]:
        # Random walk
        if random.random() < self.p:
            return random.choice(candidates), 0.0

        # Most recently flipped candidate (smallest age = most recent flip)
        most_recent = min(candidates, key=lambda v: state.age(v))

        # Min-break candidates
        breaks = {v: state.break_count(v) for v in candidates}
        min_b = min(breaks.values())
        pool = [v for v in candidates if breaks[v] == min_b]

        # Novelty: exclude the most recently flipped variable from the pool
        # unless it is the only one there
        if len(pool) > 1 and most_recent in pool:
            pool.remove(most_recent)

        return random.choice(pool), 0.0

    def is_learnable(self) -> bool:
        return False
