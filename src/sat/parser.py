"""
DIMACS .cnf parser.

Internal representation: clauses are lists of (var, polarity) tuples where var
is 0-indexed and polarity=True means the positive literal (x_var).
Also builds a per-variable adjacency list var_clauses for O(1) incremental state updates.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CNFFormula:
    n_vars: int
    n_clauses: int
    # clauses[i] = [(var, polarity), ...] — 0-indexed var, True = positive literal
    clauses: list[list[tuple[int, bool]]]
    # var_clauses[v] = [(clause_idx, polarity), ...] — which clauses contain variable v
    var_clauses: list[list[tuple[int, bool]]]


def parse_dimacs(path: str | Path) -> CNFFormula:
    """
    Parse a DIMACS .cnf file. Handles multi-line clauses and comment lines.
    Converts DIMACS 1-indexed literals to 0-indexed (var, polarity) pairs.
    """
    clauses: list[list[tuple[int, bool]]] = []
    n_vars = 0
    pending: list[int] = []  # tokens not yet terminated by 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c') or line.startswith('%'):
                continue
            if line.startswith('p'):
                parts = line.split()
                n_vars = int(parts[2])
                continue
            tokens = list(map(int, line.split()))
            pending.extend(tokens)
            # Each clause ends with a 0 terminator; may span multiple lines
            while 0 in pending:
                idx = pending.index(0)
                lits = pending[:idx]
                pending = pending[idx + 1:]
                if lits:
                    clauses.append([(abs(l) - 1, l > 0) for l in lits])

    # Build per-variable clause adjacency list
    var_clauses: list[list[tuple[int, bool]]] = [[] for _ in range(n_vars)]
    for ci, clause in enumerate(clauses):
        for (var, pol) in clause:
            var_clauses[var].append((ci, pol))

    return CNFFormula(
        n_vars=n_vars,
        n_clauses=len(clauses),
        clauses=clauses,
        var_clauses=var_clauses,
    )
