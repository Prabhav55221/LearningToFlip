"""
DIMACS .cnf parser.

Produces a list of clauses (each a list of signed integers, 1-indexed literals)
and the variable count. The rest of the codebase works with 0-indexed variables
internally; the parser handles the conversion.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CNFFormula:
    n_vars: int
    n_clauses: int
    clauses: list[list[int]]  # 0-indexed variables, sign encodes polarity


def parse_dimacs(path: str | Path) -> CNFFormula:
    """Parse a DIMACS .cnf file into a CNFFormula."""
    raise NotImplementedError
