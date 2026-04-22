"""
Incremental WalkSAT state.

Maintained incrementally after each flip (O(clause_width) work per flip):
  - n_true[c]: number of True literals in clause c
  - _unsat_set / _unsat_list / _unsat_pos: O(1) add, remove, and random sampling
    of unsatisfied clause indices via a swap-and-pop indexed list
  - unsat_deg[v]: number of unsatisfied clauses containing variable v

Computed on demand (O(degree) per variable, called only for clause candidates):
  - make_count(v): UNSAT clauses that flipping v would satisfy
  - break_count(v): SAT clauses that flipping v would violate (v is sole satisfier)
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

        self._deg = np.array([len(formula.var_clauses[v]) for v in range(n)], dtype=np.int32)

        self._n_true = np.zeros(m, dtype=np.int32)
        for ci, clause in enumerate(formula.clauses):
            self._n_true[ci] = sum(1 for (var, pol) in clause if assignment[var] == pol)

        # Swap-and-pop structure for O(1) add / discard / random-sample on unsat set.
        # _unsat_set  — membership test (O(1))
        # _unsat_list — contiguous list for O(1) random indexing
        # _unsat_pos  — clause → position in list (O(1) removal via swap-and-pop)
        self._unsat_set: set[int] = set()
        self._unsat_list: list[int] = []
        self._unsat_pos: dict[int, int] = {}
        for ci in range(m):
            if self._n_true[ci] == 0:
                self._unsat_add(ci)

        self._unsat_deg = np.zeros(n, dtype=np.int32)
        for ci in self._unsat_list:
            for (var, _) in formula.clauses[ci]:
                self._unsat_deg[var] += 1

        self._flip_count = np.zeros(n, dtype=np.int32)
        self._last_flip = np.full(n, -1, dtype=np.int64)
        self._last_policy_flip = np.full(n, -1, dtype=np.int64)
        self._policy_flip_window: list[int] = []

    # ------------------------------------------------------------------ #
    # O(1) unsat set operations                                           #
    # ------------------------------------------------------------------ #

    def _unsat_add(self, ci: int) -> None:
        if ci not in self._unsat_set:
            self._unsat_set.add(ci)
            self._unsat_pos[ci] = len(self._unsat_list)
            self._unsat_list.append(ci)

    def _unsat_discard(self, ci: int) -> None:
        if ci in self._unsat_set:
            self._unsat_set.discard(ci)
            pos = self._unsat_pos.pop(ci)
            last = self._unsat_list[-1]
            if last != ci:
                self._unsat_list[pos] = last
                self._unsat_pos[last] = pos
            self._unsat_list.pop()

    @classmethod
    def random_init(cls, formula: CNFFormula) -> "SLSState":
        assignment = np.random.randint(0, 2, size=formula.n_vars).astype(bool)
        return cls(formula, assignment)

    # ------------------------------------------------------------------ #
    # On-demand feature computation                                        #
    # ------------------------------------------------------------------ #

    def make_count(self, var: int) -> int:
        new_val = not self.assignment[var]
        return sum(
            1 for (ci, pol) in self.formula.var_clauses[var]
            if ci in self._unsat_set and pol == new_val
        )

    def break_count(self, var: int) -> int:
        old_val = bool(self.assignment[var])
        return sum(
            1 for (ci, pol) in self.formula.var_clauses[var]
            if ci not in self._unsat_set and pol == old_val and self._n_true[ci] == 1
        )

    def age(self, var: int) -> int:
        lf = int(self._last_flip[var])
        return self.step if lf < 0 else self.step - lf

    def policy_age(self, var: int) -> int:
        lf = int(self._last_policy_flip[var])
        return self.step if lf < 0 else self.step - lf

    def in_last_k_policy(self, var: int, k: int) -> bool:
        return var in self._policy_flip_window[:k]

    # ------------------------------------------------------------------ #
    # Incremental properties                                               #
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
        make_c = self.make_count(var)
        break_c = self.break_count(var)

        old_val = bool(self.assignment[var])
        new_val = not old_val
        self.assignment[var] = new_val

        for (ci, pol) in self.formula.var_clauses[var]:
            lit_was_true = (pol == old_val)
            if lit_was_true:
                self._n_true[ci] -= 1
                if self._n_true[ci] == 0:
                    self._unsat_add(ci)
                    for (u, _) in self.formula.clauses[ci]:
                        self._unsat_deg[u] += 1
            else:
                self._n_true[ci] += 1
                if self._n_true[ci] == 1:
                    self._unsat_discard(ci)
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
        """O(1) random unsatisfied clause selection via indexed list."""
        ci = self._unsat_list[random.randrange(len(self._unsat_list))]
        return [var for (var, _) in self.formula.clauses[ci]]

    @property
    def n_unsat(self) -> int:
        return len(self._unsat_list)

    @property
    def is_solved(self) -> bool:
        return len(self._unsat_list) == 0
