"""
Shared fixtures.

SMALL_CNF is a 3-variable, 4-clause formula used across all unit tests.
With assignment (F, F, F) the initial satisfaction is known exactly:

    Clause 0: ( x0  v  x1  v  x2 ) -> UNSAT  n_true=0
    Clause 1: (¬x0  v  x1  v  x2 ) -> SAT    n_true=1  (¬x0 is the sole satisfier)
    Clause 2: ( x0  v ¬x1  v  x2 ) -> SAT    n_true=1  (¬x1 is the sole satisfier)
    Clause 3: ( x0  v  x1  v ¬x2 ) -> SAT    n_true=1  (¬x2 is the sole satisfier)
"""

import textwrap
import numpy as np
import pytest
from src.sat.parser import parse_dimacs, CNFFormula
from src.sat.state import SLSState


SMALL_DIMACS = textwrap.dedent("""\
    p cnf 3 4
    1 2 3 0
    -1 2 3 0
    1 -2 3 0
    1 2 -3 0
""")


@pytest.fixture
def small_formula(tmp_path) -> CNFFormula:
    f = tmp_path / "small.cnf"
    f.write_text(SMALL_DIMACS)
    return parse_dimacs(f)


@pytest.fixture
def all_false_state(small_formula) -> SLSState:
    """SLSState for SMALL_DIMACS with assignment (F, F, F)."""
    assignment = np.zeros(small_formula.n_vars, dtype=bool)
    return SLSState(small_formula, assignment)
