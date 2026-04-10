#!/usr/bin/env python3
"""
Check if a DIMACS CNF file is satisfiable using PySAT (Glucose3).
Exit 0 if SAT, exit 1 if UNSAT.

Used by generate_data.sh to filter satisfiable instances.
"""
import sys

try:
    from pysat.formula import CNF
    from pysat.solvers import Glucose3
except ImportError:
    print("ERROR: python-sat not installed. Run: pip install python-sat", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: check_sat.py <file.cnf>", file=sys.stderr)
        sys.exit(2)

    formula = CNF(from_file=sys.argv[1])
    with Glucose3(bootstrap_with=formula.clauses) as solver:
        sat = solver.solve()

    sys.exit(0 if sat else 1)


if __name__ == "__main__":
    main()
