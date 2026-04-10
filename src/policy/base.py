"""
Policy protocol — the single interface the SLS loop depends on.

Every policy (classical or learned) implements select(). The SLS loop calls it
with the candidate variables in the selected unsatisfied clause and the current
state, and uses the returned variable as the next flip.

Classical baselines implement their own selection logic directly.
Learned policies (linear, MLP) sample from a softmax distribution and return
the log-probability for use in REINFORCE training.
"""

from typing import Protocol, runtime_checkable
from src.sat.state import SLSState


@runtime_checkable
class Policy(Protocol):
    def select(self, candidates: list[int], state: SLSState) -> tuple[int, float, bool]:
        """
        Choose a variable from candidates to flip.

        Returns:
            var:       the chosen variable (must be in candidates)
            log_prob:  log probability of the choice (0.0 for classical baselines)
            by_policy: False when the flip is a noise/random-walk step that should
                       NOT update age2 (policy recency). True for all classical
                       baselines and for scored selections in learned policies.
        """
        ...

    def is_learnable(self) -> bool:
        """True for policies with trainable parameters."""
        return False
