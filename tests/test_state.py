"""
Tests for SLSState incremental updates.

All expected values are derived analytically from SMALL_DIMACS with (F,F,F):

    Clause 0: ( x0  v  x1  v  x2 )  UNSAT  n_true=0
    Clause 1: (¬x0  v  x1  v  x2 )  SAT    n_true=1
    Clause 2: ( x0  v ¬x1  v  x2 )  SAT    n_true=1
    Clause 3: ( x0  v  x1  v ¬x2 )  SAT    n_true=1
"""

import numpy as np
import pytest
from src.sat.state import SLSState


# ------------------------------------------------------------------ #
# Initialisation                                                       #
# ------------------------------------------------------------------ #

def test_n_true_on_init(all_false_state):
    s = all_false_state
    assert s._n_true[0] == 0   # clause 0 UNSAT
    assert s._n_true[1] == 1   # ¬x0 is True
    assert s._n_true[2] == 1   # ¬x1 is True
    assert s._n_true[3] == 1   # ¬x2 is True


def test_unsat_set_on_init(all_false_state):
    assert all_false_state._unsat_set == {0}


def test_n_unsat_on_init(all_false_state):
    assert all_false_state.n_unsat == 1


def test_unsat_deg_on_init(all_false_state):
    # All 3 variables appear in clause 0 (the only UNSAT clause)
    assert all(all_false_state.unsat_deg == 1)


def test_deg_on_init(all_false_state):
    # Each variable appears in all 4 clauses
    assert all(all_false_state.deg == 4)


def test_flip_count_zero_on_init(all_false_state):
    assert all(all_false_state.flip_count == 0)


def test_age_before_any_flip(all_false_state):
    # age returns current step (0) if never flipped
    assert all_false_state.age(0) == 0
    assert all_false_state.age(1) == 0
    assert all_false_state.age(2) == 0


# ------------------------------------------------------------------ #
# On-demand make / break                                               #
# ------------------------------------------------------------------ #

def test_make_count_x0(all_false_state):
    # Flipping x0 (F→T): positive x0 literal in clause 0 (UNSAT) becomes True → make=1
    assert all_false_state.make_count(0) == 1


def test_break_count_x0(all_false_state):
    # Flipping x0 (F→T): ¬x0 in clause 1 is the sole satisfier (n_true=1) → break=1
    assert all_false_state.break_count(0) == 1


def test_make_count_symmetric(all_false_state):
    # By symmetry, x1 and x2 have identical make/break to x0
    assert all_false_state.make_count(1) == 1
    assert all_false_state.make_count(2) == 1


def test_break_count_symmetric(all_false_state):
    assert all_false_state.break_count(1) == 1
    assert all_false_state.break_count(2) == 1


# ------------------------------------------------------------------ #
# Flip: incremental state update                                       #
# ------------------------------------------------------------------ #

def test_flip_returns_correct_make_break(all_false_state):
    make_c, break_c = all_false_state.flip(0)
    assert make_c == 1
    assert break_c == 1


def test_flip_x0_updates_assignment(all_false_state):
    all_false_state.flip(0)
    assert all_false_state.assignment[0] is True or all_false_state.assignment[0] == True


def test_flip_x0_updates_n_true(all_false_state):
    all_false_state.flip(0)
    # Clause 0 ( x0 v x1 v x2): x0=T → n_true 0→1
    assert all_false_state._n_true[0] == 1
    # Clause 1 (¬x0 v x1 v x2): ¬x0=F → n_true 1→0
    assert all_false_state._n_true[1] == 0
    # Clause 2 ( x0 v ¬x1 v x2): x0=T → n_true 1→2
    assert all_false_state._n_true[2] == 2
    # Clause 3 ( x0 v x1 v ¬x2): x0=T → n_true 1→2
    assert all_false_state._n_true[3] == 2


def test_flip_x0_updates_unsat_set(all_false_state):
    all_false_state.flip(0)
    # Clause 0 becomes SAT, clause 1 becomes UNSAT
    assert 0 not in all_false_state._unsat_set
    assert 1 in all_false_state._unsat_set
    assert all_false_state.n_unsat == 1


def test_flip_x0_updates_step(all_false_state):
    all_false_state.flip(0)
    assert all_false_state.step == 1


def test_flip_x0_updates_flip_count(all_false_state):
    all_false_state.flip(0)
    assert all_false_state.flip_count[0] == 1
    assert all_false_state.flip_count[1] == 0


def test_age_after_flip(all_false_state):
    all_false_state.flip(0)
    # x0 was just flipped at step 0; current step is now 1 → age = 1
    assert all_false_state.age(0) == 1
    # x1 never flipped; age = current step = 1
    assert all_false_state.age(1) == 1


def test_not_solved_after_partial_flip(all_false_state):
    all_false_state.flip(0)
    assert not all_false_state.is_solved


# ------------------------------------------------------------------ #
# Solve by hand                                                        #
# ------------------------------------------------------------------ #

def test_manual_solve(all_false_state):
    """Flipping x1 after x0 should solve the formula."""
    s = all_false_state
    s.flip(0)   # Now clause 1 is UNSAT: (¬x0 v x1 v x2) with x0=T, x1=F, x2=F
    s.flip(1)   # x1=T satisfies clause 1
    assert s.is_solved
