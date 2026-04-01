"""
Policy protocol — the single interface the SLS loop depends on.

Every policy (classical or learned) implements score(). The SLS loop calls it
with the candidate variables in the selected unsatisfied clause and the current
state, then samples from the returned distribution.
"""

from typing import Protocol, runtime_checkable
import numpy as np
from src.sat.state import SLSState


@runtime_checkable
class Policy(Protocol):
    def score(self, candidates: list[int], state: SLSState) -> np.ndarray:
        """
        Return an unnormalised score for each candidate variable.
        Shape: (len(candidates),). Higher = more preferred.
        The SLS loop applies softmax and samples.
        """
        ...

    def is_learnable(self) -> bool:
        """True for policies with trainable parameters."""
        return False
