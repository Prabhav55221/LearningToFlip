"""
Feature extraction: φ(x, state) → feature vector.

Two families of features:

  Interian features (exact replication of Interian & Bernardini KR 2023):
    bk_log       log(min(break(x), 5) + 1) / log(6)  — normalized to [0,1]
    policy_last10 1 if var was among last 10 policy-selected variables
    policy_last5  1 if var was among last 5 policy-selected variables
    delta1        age(x) / t   — any-flip recency, normalized by step
    delta2        policy_age(x) / t  — policy-flip recency, normalized by step

  Our extended features (9 total, used in ablation sets):
    break, make, age, is_recent_5, is_recent_10,
    break_zero, unsat_deg, deg, flip_count

All values are cached within a single call where multiple features share
the same underlying computation.
"""

import math
import numpy as np
from src.sat.state import SLSState


# --- Interian & Bernardini (KR 2023) exact feature set ---
# Order matches their stats_per_clause: [breaks, in_last_10, in_last_5, age, age2]
INTERIAN_FEATURES = [
    "bk_log",        # log(min(break, 5) + 1) / log(6) — normalized to [0,1]
    "policy_last10", # 1 if var in last 10 policy-selected variables
    "policy_last5",  # 1 if var in last 5 policy-selected variables
    "delta1",        # any-flip recency normalized by step
    "delta2",        # policy-flip recency normalized by step
]

# --- Our extended feature set (9 features for ablation) ---
ALL_FEATURES = [
    "break",
    "make",
    "age",
    "is_recent_5",
    "is_recent_10",
    "break_zero",
    "unsat_deg",
    "deg",
    "flip_count",
]

FEATURE_SETS = {
    "interian":   INTERIAN_FEATURES,
    "base":       ["make", "break", "age", "is_recent_5", "is_recent_10"],
    "full":       ALL_FEATURES,
    "no_recency": ["make", "break", "age", "break_zero", "unsat_deg", "deg", "flip_count"],
}


def extract(var: int, state: SLSState, feature_set: str = "full") -> np.ndarray:
    """
    Extract the feature vector for variable var in the given state.
    Returns shape (len(features),) float32 array.
    """
    return extract_named(var, state, FEATURE_SETS[feature_set])


def extract_named(var: int, state: SLSState, names: list[str]) -> np.ndarray:
    """Extract a specific list of features by name. Caches break/make/age per call."""
    # Lazy-compute values that may be needed by multiple features
    _brk: int | None = None
    _mk: int | None = None
    _age: int | None = None
    _policy_age: int | None = None

    def brk() -> int:
        nonlocal _brk
        if _brk is None:
            _brk = state.break_count(var)
        return _brk

    def mk() -> int:
        nonlocal _mk
        if _mk is None:
            _mk = state.make_count(var)
        return _mk

    def age() -> int:
        nonlocal _age
        if _age is None:
            _age = state.age(var)
        return _age

    def policy_age() -> int:
        nonlocal _policy_age
        if _policy_age is None:
            _policy_age = state.policy_age(var)
        return _policy_age

    t = state.step  # current step for normalization

    vec = []
    for name in names:
        # --- Interian & Bernardini (KR 2023) features ---
        if name == "bk_log":
            # log(min(break, 5) + 1) / log(6): normalized to [0, 1]
            vec.append(float(math.log(min(brk(), 5) + 1) / math.log(6)))
        elif name == "delta1":
            # Δ1 = 1 - age1/t = age(x)/t  (any-flip recency, normalized)
            vec.append(age() / t if t > 0 else 0.0)
        elif name == "delta2":
            # Δ2 = 1 - age2/t = policy_age(x)/t  (policy-flip recency, normalized)
            vec.append(policy_age() / t if t > 0 else 0.0)
        elif name == "policy_last10":
            vec.append(float(state.in_last_k_policy(var, 10)))
        elif name == "policy_last5":
            vec.append(float(state.in_last_k_policy(var, 5)))
        # --- Our extended features ---
        elif name == "break":
            vec.append(float(brk()))
        elif name == "make":
            vec.append(float(mk()))
        elif name == "age":
            vec.append(float(age()))
        elif name == "is_recent_5":
            vec.append(float(age() <= 5))
        elif name == "is_recent_10":
            vec.append(float(age() <= 10))
        elif name == "break_zero":
            vec.append(float(brk() == 0))
        elif name == "unsat_deg":
            vec.append(float(state.unsat_deg[var]))
        elif name == "deg":
            vec.append(float(state.deg[var]))
        elif name == "flip_count":
            vec.append(float(state.flip_count[var]))
        else:
            raise ValueError(f"Unknown feature: {name}")

    return np.array(vec, dtype=np.float32)


def extract_batch(candidates: list[int], state: SLSState, feature_set: str = "full") -> np.ndarray:
    """
    Extract features for all candidates at once.
    Returns shape (len(candidates), n_features) float32 array.
    """
    return np.stack([extract(v, state, feature_set) for v in candidates])
