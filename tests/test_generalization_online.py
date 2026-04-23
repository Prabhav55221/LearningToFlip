"""Smoke tests for the rewritten generalization and online-adaptation code."""

import torch

from scripts.eval_generalization import infer_mlp_architecture
from src.policy.mlp import MLPPolicy
from src.sat.parser import parse_dimacs
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
