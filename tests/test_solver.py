"""End-to-end tests for the SLS solver loop."""

import textwrap
import numpy as np
import pytest
from src.sat.parser import parse_dimacs
from src.sat.state import SLSState
from src.policy.baselines import MinBreak, NoveltyPlus
from src.sls.solver import run_try, solve, StepRecord


# ------------------------------------------------------------------ #
# Trivial formula: one clause (x0), solved by one flip if x0=F        #
# ------------------------------------------------------------------ #

TRIVIAL_DIMACS = "p cnf 1 1\n1 0\n"


@pytest.fixture
def trivial_formula(tmp_path):
    f = tmp_path / "trivial.cnf"
    f.write_text(TRIVIAL_DIMACS)
    return parse_dimacs(f)


# ------------------------------------------------------------------ #
# run_try                                                              #
# ------------------------------------------------------------------ #

class TestRunTry:
    def test_solves_trivial_formula(self, trivial_formula):
        result = run_try(trivial_formula, MinBreak(), max_flips=10)
        assert result.solved
        assert result.n_flips <= 1   # at most one flip needed

    def test_result_fields(self, trivial_formula):
        result = run_try(trivial_formula, MinBreak(), max_flips=10)
        assert isinstance(result.solved, bool)
        assert isinstance(result.n_flips, int)
        assert result.n_flips >= 0

    def test_trajectory_empty_when_not_recorded(self, trivial_formula):
        result = run_try(trivial_formula, MinBreak(), max_flips=10,
                         record_trajectory=False)
        assert result.trajectory == []

    def test_trajectory_populated_when_recorded(self, small_formula):
        result = run_try(small_formula, MinBreak(), max_flips=50,
                         record_trajectory=True)
        if result.solved:
            assert len(result.trajectory) == result.n_flips
        for rec in result.trajectory:
            assert isinstance(rec, StepRecord)
            assert rec.log_prob == 0.0  # MinBreak is classical
            assert isinstance(rec.reward, float)

    def test_returns_unsolved_when_budget_exhausted(self, small_formula):
        # Budget of 0 means it can't even start flipping
        result = run_try(small_formula, MinBreak(), max_flips=0)
        assert not result.solved
        assert result.n_flips == 0


# ------------------------------------------------------------------ #
# solve (multi-try)                                                    #
# ------------------------------------------------------------------ #

class TestSolve:
    def test_solves_trivial_with_one_try(self, trivial_formula):
        result = solve(trivial_formula, MinBreak(), max_flips=10, max_tries=1)
        assert result.solved

    def test_returns_first_success(self, trivial_formula):
        result = solve(trivial_formula, MinBreak(), max_flips=10, max_tries=5)
        assert result.solved

    def test_noveltyplus_also_solves(self, small_formula):
        result = solve(small_formula, NoveltyPlus(p=0.1),
                       max_flips=500, max_tries=10)
        assert result.solved

    def test_minbreak_solves_small(self, small_formula):
        result = solve(small_formula, MinBreak(),
                       max_flips=500, max_tries=10)
        assert result.solved
