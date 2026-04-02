"""Tests for DIMACS parser."""

import textwrap
import pytest
from src.sat.parser import parse_dimacs


def test_basic_parse(small_formula):
    assert small_formula.n_vars == 3
    assert small_formula.n_clauses == 4


def test_clause_contents(small_formula):
    # Clause 0: 1 2 3 0  ->  (x0 T), (x1 T), (x2 T)
    assert small_formula.clauses[0] == [(0, True), (1, True), (2, True)]
    # Clause 1: -1 2 3 0  -> (x0 F), (x1 T), (x2 T)
    assert small_formula.clauses[1] == [(0, False), (1, True), (2, True)]


def test_var_clauses_adjacency(small_formula):
    # x0 appears in all 4 clauses
    clause_ids = [ci for (ci, _) in small_formula.var_clauses[0]]
    assert sorted(clause_ids) == [0, 1, 2, 3]


def test_var_clauses_polarity(small_formula):
    # x0 appears positive in clauses 0,2,3 and negative in clause 1
    pol_by_clause = {ci: pol for (ci, pol) in small_formula.var_clauses[0]}
    assert pol_by_clause[0] is True
    assert pol_by_clause[1] is False
    assert pol_by_clause[2] is True
    assert pol_by_clause[3] is True


def test_multiline_clause(tmp_path):
    # DIMACS allows a clause to span multiple lines before the 0 terminator
    content = "p cnf 2 1\n1\n2 0\n"
    f = tmp_path / "multi.cnf"
    f.write_text(content)
    formula = parse_dimacs(f)
    assert formula.n_clauses == 1
    assert formula.clauses[0] == [(0, True), (1, True)]


def test_comment_lines_ignored(tmp_path):
    content = "c this is a comment\np cnf 2 1\nc another comment\n1 2 0\n"
    f = tmp_path / "comments.cnf"
    f.write_text(content)
    formula = parse_dimacs(f)
    assert formula.n_vars == 2
    assert formula.n_clauses == 1
