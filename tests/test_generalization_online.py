"""Smoke tests for the rewritten generalization and online-adaptation code."""

import torch

import scripts.eval_generalization as generalization
from scripts.eval_generalization import infer_mlp_architecture
from src.policy.mlp import MLPPolicy
from src.sat.parser import parse_dimacs
from src.sls.solver import SolveResult
from src.train.online import (
    OnlineKLAdapter,
    OnlineKLConfig,
    OnlineSuccessKLAdapter,
    OnlineSuccessKLConfig,
)


TRIVIAL_DIMACS = "p cnf 1 1\n1 0\n"


def test_infer_mlp_architecture(tmp_path):
    model = MLPPolicy(feature_set="full", hidden_dim=32, n_layers=1)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    hidden_dim, n_layers = infer_mlp_architecture(path)
    assert hidden_dim == 32
    assert n_layers == 1


def _trivial_formula(tmp_path):
    path = tmp_path / "trivial.cnf"
    path.write_text(TRIVIAL_DIMACS)
    return parse_dimacs(path)


def test_online_kl_adapter_solves_trivial_formula(tmp_path):
    formula = _trivial_formula(tmp_path)
    policy = MLPPolicy(feature_set="full", hidden_dim=8, n_layers=1)
    adapter = OnlineKLAdapter(policy, OnlineKLConfig())

    result = adapter.solve(formula, max_flips=10, max_tries=2)
    assert result.solved
    assert result.n_flips <= 1


def test_online_success_kl_adapter_solves_trivial_formula(tmp_path):
    formula = _trivial_formula(tmp_path)
    policy = MLPPolicy(feature_set="full", hidden_dim=8, n_layers=1)
    adapter = OnlineSuccessKLAdapter(policy, OnlineSuccessKLConfig())

    result = adapter.solve(formula, max_flips=10, max_tries=2)
    assert result.solved
    assert result.n_flips <= 1


def test_online_adapters_reset_to_offline(tmp_path):
    _trivial_formula(tmp_path)
    policy = MLPPolicy(feature_set="full", hidden_dim=8, n_layers=1)
    offline_state = {k: v.clone() for k, v in policy.state_dict().items()}

    adapter = OnlineKLAdapter(policy, OnlineKLConfig())
    for param in policy.parameters():
        param.data.add_(torch.ones_like(param))
    adapter.reset_to_offline()

    for key, value in offline_state.items():
        assert torch.allclose(policy.state_dict()[key], value)

    success_policy = MLPPolicy(feature_set="full", hidden_dim=8, n_layers=1)
    success_offline = {k: v.clone() for k, v in success_policy.state_dict().items()}
    success_adapter = OnlineSuccessKLAdapter(success_policy, OnlineSuccessKLConfig())
    for param in success_policy.parameters():
        param.data.add_(torch.ones_like(param))
    success_adapter.reset_to_offline()

    for key, value in success_offline.items():
        assert torch.allclose(success_policy.state_dict()[key], value)


def test_eval_restart_policy_uses_all_tries_and_reports_cumulative(monkeypatch):
    calls = []
    outcomes = [
        SolveResult(solved=False, n_flips=10),
        SolveResult(solved=True, n_flips=4),
        SolveResult(solved=True, n_flips=7),
    ]

    def fake_run_try(formula, policy, max_flips):
        calls.append((formula, policy, max_flips))
        return outcomes[len(calls) - 1]

    monkeypatch.setattr(generalization, "run_try", fake_run_try)

    result = generalization.eval_restart_policy(
        policy=object(),
        formula=object(),
        max_flips=10,
        max_tries=3,
    )

    assert len(calls) == 3
    assert result == {
        "solved": True,
        "best_flips": 4,
        "cumulative_flips": 21,
        "n_tries": 3,
    }


def test_online_success_adapter_evaluate_reports_cumulative_effort(tmp_path, monkeypatch):
    formula = _trivial_formula(tmp_path)
    policy = MLPPolicy(feature_set="full", hidden_dim=8, n_layers=1)
    adapter = OnlineSuccessKLAdapter(policy, OnlineSuccessKLConfig())

    outcomes = iter([
        (False, 10, []),
        (True, 3, [("phi", 0)]),
        (True, 5, [("phi", 0)]),
    ])
    fine_tunes = []

    def fake_run_try(formula, max_flips):
        return next(outcomes)

    def fake_fine_tune(trajectory):
        fine_tunes.append(len(trajectory))
        return 0.0

    monkeypatch.setattr(adapter, "_run_try", fake_run_try)
    monkeypatch.setattr(adapter, "_fine_tune", fake_fine_tune)

    result = adapter.evaluate(formula, max_flips=10, max_tries=3, reset=False)

    assert result.solved is True
    assert result.best_flips == 3
    assert result.cumulative_flips == 18
    assert result.n_tries == 3
    assert fine_tunes == [1, 1]
