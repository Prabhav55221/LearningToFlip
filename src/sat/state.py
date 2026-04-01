"""
Incremental WalkSAT state.

All per-variable statistics (make, break, unsat_deg, age, flip_count) are
maintained incrementally — O(clause_width) work per flip, O(1) feature lookup.

When variable x is flipped:
  - Iterate over clauses containing x
  - For each clause that changes satisfaction status, update make/break/unsat_deg
    for all variables sharing that clause
"""

import random
import numpy as np
from dataclasses import dataclass, field
from src.sat.parser import CNFFormula


@dataclass
class SLSState:
    formula: CNFFormula
    assignment: np.ndarray        # shape (n_vars,), dtype bool
    step: int = 0

    # Per-variable incremental statistics
    make: np.ndarray = field(init=False)       # make[x]: unsat clauses x would satisfy
    break_: np.ndarray = field(init=False)     # break_[x]: sat clauses x would violate
    unsat_deg: np.ndarray = field(init=False)  # unsat_deg[x]: unsat clauses containing x
    deg: np.ndarray = field(init=False)        # deg[x]: total clauses containing x (constant)
    flip_count: np.ndarray = field(init=False) # flip_count[x]: total flips this episode
    last_flip: np.ndarray = field(init=False)  # last_flip[x]: step of last flip (-1 if never)

    # Clause-level tracking
    n_satisfied: int = field(init=False)
    unsat_clause_indices: set = field(init=False)

    def __post_init__(self) -> None:
        raise NotImplementedError

    @classmethod
    def random_init(cls, formula: CNFFormula) -> "SLSState":
        """Initialise with a uniformly random assignment."""
        assignment = np.random.randint(0, 2, size=formula.n_vars, dtype=bool)
        state = cls(formula=formula, assignment=assignment)
        return state

    def flip(self, var: int) -> tuple[int, int]:
        """
        Flip variable var. Returns (make_count, break_count) for the flip,
        which together give the reward r_t = make - break.
        Updates all incremental statistics.
        """
        raise NotImplementedError

    def random_unsat_clause(self) -> list[int]:
        """Sample a uniformly random unsatisfied clause. Returns list of variables."""
        raise NotImplementedError

    @property
    def n_unsat(self) -> int:
        return len(self.unsat_clause_indices)

    @property
    def is_solved(self) -> bool:
        return self.n_unsat == 0

    def age(self, var: int) -> int:
        """Steps since var was last flipped. Returns step if never flipped."""
        if self.last_flip[var] < 0:
            return self.step
        return self.step - self.last_flip[var]
