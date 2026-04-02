"""Tests for MinBreak and NoveltyPlus baselines."""

import random
import pytest
from src.policy.baselines import MinBreak, NoveltyPlus


# ------------------------------------------------------------------ #
# MinBreak                                                             #
# ------------------------------------------------------------------ #

class TestMinBreak:
    def test_select_returns_valid_variable(self, all_false_state):
        """Selected variable must be one of the candidates."""
        candidates = all_false_state.random_unsat_clause()
        policy = MinBreak()
        var, log_prob = policy.select(candidates, all_false_state)
        assert var in candidates

    def test_log_prob_is_zero(self, all_false_state):
        candidates = all_false_state.random_unsat_clause()
        policy = MinBreak()
        _, log_prob = policy.select(candidates, all_false_state)
        assert log_prob == 0.0

    def test_selects_min_break(self, all_false_state):
        """
        All 3 candidates have break=1 by symmetry in SMALL_DIMACS (F,F,F).
        Any of them is valid; the policy should at least not exceed min break.
        """
        candidates = all_false_state.random_unsat_clause()
        policy = MinBreak()
        var, _ = policy.select(candidates, all_false_state)
        chosen_break = all_false_state.break_count(var)
        min_break = min(all_false_state.break_count(v) for v in candidates)
        assert chosen_break == min_break

    def test_tiebreak_explores_all_candidates(self, all_false_state):
        """When all candidates are tied, every candidate should be chosen eventually."""
        candidates = all_false_state.random_unsat_clause()
        policy = MinBreak()
        seen = set()
        for _ in range(200):
            var, _ = policy.select(candidates, all_false_state)
            seen.add(var)
            if seen == set(candidates):
                break
        assert seen == set(candidates), "Tiebreaking should explore all tied candidates"

    def test_is_not_learnable(self):
        assert MinBreak().is_learnable() is False


# ------------------------------------------------------------------ #
# NoveltyPlus                                                          #
# ------------------------------------------------------------------ #

class TestNoveltyPlus:
    def test_select_returns_valid_variable(self, all_false_state):
        candidates = all_false_state.random_unsat_clause()
        policy = NoveltyPlus(p=0.1)
        var, _ = policy.select(candidates, all_false_state)
        assert var in candidates

    def test_log_prob_is_zero(self, all_false_state):
        candidates = all_false_state.random_unsat_clause()
        policy = NoveltyPlus()
        _, log_prob = policy.select(candidates, all_false_state)
        assert log_prob == 0.0

    def test_random_walk_p1_always_random(self, all_false_state):
        """With p=1.0, every selection is pure random walk — all candidates reachable."""
        candidates = all_false_state.random_unsat_clause()
        policy = NoveltyPlus(p=1.0)
        seen = set()
        for _ in range(300):
            var, _ = policy.select(candidates, all_false_state)
            seen.add(var)
            if seen == set(candidates):
                break
        assert seen == set(candidates)

    def test_novelty_excludes_most_recent(self, all_false_state):
        """
        After flipping x0, x0 is the most recently flipped. With p=0 and
        multiple min-break candidates, Novelty should avoid picking x0.
        """
        s = all_false_state
        s.flip(0)  # x0 is now the most recently flipped

        # Clause 1 is now the UNSAT clause: candidates are x0, x1, x2
        candidates = s.random_unsat_clause()
        if len(candidates) < 2 or 0 not in candidates:
            pytest.skip("Need x0 in candidates with alternatives for this test")

        policy = NoveltyPlus(p=0.0)  # pure Novelty, no random walk
        chosen = {policy.select(candidates, s)[0] for _ in range(100)}
        # x0 (most recently flipped) should be excluded when alternatives exist
        assert 0 not in chosen

    def test_is_not_learnable(self):
        assert NoveltyPlus().is_learnable() is False
