"""
Classical SLS baselines. No learnable parameters.

Both implement the Policy protocol and can be dropped into the SLS loop
or evaluation runner identically to the learned policies.
"""

import numpy as np
from src.sat.state import SLSState


class MinBreak:
    """
    WalkSAT min-break: score = -break(x).
    Ties broken uniformly at random by the SLS loop's softmax sampling.
    """

    def score(self, candidates: list[int], state: SLSState) -> np.ndarray:
        return -state.break_[candidates].astype(np.float32)

    def is_learnable(self) -> bool:
        return False


class NoveltyPlus:
    """
    Novelty+: avoid the most recently flipped variable unless it is the only
    candidate with break == 0; add random walk with probability p.

    Novelty rule: among candidates with min break, skip the most recently
    flipped unless all others have strictly higher break.
    Novelty+: with probability p, instead pick a random candidate (random walk).
    """

    def __init__(self, p: float = 0.1) -> None:
        self.p = p

    def score(self, candidates: list[int], state: SLSState) -> np.ndarray:
        raise NotImplementedError

    def is_learnable(self) -> bool:
        return False
