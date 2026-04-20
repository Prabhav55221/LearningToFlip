"""
Incremental WalkSAT state.

Maintained incrementally after each flip (O(clause_width) work per flip):
  - n_true[c]: number of True literals in clause c
  - _unsat: set of unsatisfied clause indices (for random clause selection)
  - unsat_deg[v]: number of unsatisfied clauses containing variable v

Computed on demand (O(degree) per variable, called only for clause candidates):
  - make_count(v): UNSAT clauses that flipping v would satisfy
  - break_count(v): SAT clauses that flipping v would violate (v is sole satisfier)

This avoids the complexity of fully incremental make/break while staying fast
enough at our scales (n=100-200, degree ~13 for 3-SAT at phase transition).
"""

import random
import numpy as np
from src.sat.parser import CNFFormula


class SLSState:
    def __init__(self, formula: CNFFormula, assignment: np.ndarray) -> None:
        self.formula = formula
        self.assignment = assignment.copy().astype(bool)
        self.step: int = 0

        n = formula.n_vars
        m = formula.n_clauses

        # Precomputed: total clause degree per variable (constant)
        self._deg = np.array([len(formula.var_clauses[v]) for v in range(n)], dtype=np.int32)

        # n_true[c]: number of currently-True literals in clause c
        self._n_true = np.zeros(m, dtype=np.int32)
        for ci, clause in enumerate(formula.clauses):
            self._n_true[ci] = sum(1 for (var, pol) in clause if assignment[var] == pol)

        # Unsatisfied clause index set
        self._unsat: set[int] = {ci for ci in range(m) if self._n_true[ci] == 0}

        # unsat_deg[v]: number of unsatisfied clauses containing v
        self._unsat_deg = np.zeros(n, dtype=np.int32)
        for ci in self._unsat:
            for (var, _) in formula.clauses[ci]:
                self._unsat_deg[var] += 1

        # Flip history
        self._flip_count = np.zeros(n, dtype=np.int32)
        self._last_flip = np.full(n, -1, dtype=np.int64)
        # Policy-flip history (age2 / last_10 in Interian et al.):
        #   _last_policy_flip: step index of last policy-flip (for delta2 feature)
        #   _policy_flip_window: ordered list of last 10 policy-selected variables,
        #       most recent first — mirrors Interian's `last_10` list exactly.
        self._last_policy_flip = np.full(n, -1, dtype=np.int64)
        self._policy_flip_window: list[int] = []  # max length 10

    @classmethod
    def random_init(cls, formula: CNFFormula) -> "SLSState":
        assignment = np.random.randint(0, 2, size=formula.n_vars).astype(bool)
        return cls(formula, assignment)

    # ------------------------------------------------------------------ #
    # On-demand feature computation (called only for ~3 candidates/step)  #
    # ------------------------------------------------------------------ #

    def make_count(self, var: int) -> int:
        """UNSAT clauses that flipping var would satisfy."""
        new_val = not self.assignment[var]
        return sum(
            1 for (ci, pol) in self.formula.var_clauses[var]
            if ci in self._unsat and pol == new_val
        )

    def break_count(self, var: int) -> int:
        """SAT clauses that flipping var would violate (var is their sole satisfier)."""
        old_val = bool(self.assignment[var])
        return sum(
            1 for (ci, pol) in self.formula.var_clauses[var]
            if ci not in self._unsat and pol == old_val and self._n_true[ci] == 1
        )

    def age(self, var: int) -> int:
        """Steps since var was last flipped (any flip); returns current step if never flipped."""
        lf = int(self._last_flip[var])
        return self.step if lf < 0 else self.step - lf

    def policy_age(self, var: int) -> int:
        """Steps since var was last flipped by the policy (by_policy=True flips only).
        Returns current step if the variable has never been policy-flipped."""
        lf = int(self._last_policy_flip[var])
        return self.step if lf < 0 else self.step - lf

    def in_last_k_policy(self, var: int, k: int) -> bool:
        """True if var appears in the last k policy-selected variables.
        Mirrors Interian's `v in self.last_10[:k]` check exactly."""
        return var in self._policy_flip_window[:k]

    # ------------------------------------------------------------------ #
    # Incremental properties (O(1) lookup)                                #
    # ------------------------------------------------------------------ #

    @property
    def deg(self) -> np.ndarray:
        return self._deg

    @property
    def unsat_deg(self) -> np.ndarray:
        return self._unsat_deg

    @property
    def flip_count(self) -> np.ndarray:
        return self._flip_count

    # ------------------------------------------------------------------ #
    # Core operations                                                      #
    # ------------------------------------------------------------------ #

    def flip(self, var: int, by_policy: bool = True) -> tuple[int, int]:
        """
        Flip variable var. Returns (make_count, break_count) for this flip.
        Updates n_true, unsat set, unsat_deg, and recency tracking.
        """
        make_c = self.make_count(var)
        break_c = self.break_count(var)

        old_val = bool(self.assignment[var])
        new_val = not old_val
        self.assignment[var] = new_val

        for (ci, pol) in self.formula.var_clauses[var]:
            lit_was_true = (pol == old_val)
            if lit_was_true:
                # Literal True → False: clause may go SAT → UNSAT
                self._n_true[ci] -= 1
                if self._n_true[ci] == 0:
                    self._unsat.add(ci)
                    for (u, _) in self.formula.clauses[ci]:
                        self._unsat_deg[u] += 1
            else:
                # Literal False → True: clause may go UNSAT → SAT
                self._n_true[ci] += 1
                if self._n_true[ci] == 1:
                    self._unsat.discard(ci)
                    for (u, _) in self.formula.clauses[ci]:
                        self._unsat_deg[u] -= 1

        self._flip_count[var] += 1
        self._last_flip[var] = self.step
        if by_policy:
            self._last_policy_flip[var] = self.step
            self._policy_flip_window.insert(0, var)
            if len(self._policy_flip_window) > 10:
                self._policy_flip_window.pop()
        self.step += 1

        return make_c, break_c

    def random_unsat_clause(self) -> list[int]:
        """Return variable indices of a uniformly random unsatisfied clause."""
        ci = random.choice(list(self._unsat))
        return [var for (var, _) in self.formula.clauses[ci]]

    @property
    def n_unsat(self) -> int:
        return len(self._unsat)

    @property
    def is_solved(self) -> bool:
        return len(self._unsat) == 0
