"""
Feature extraction: φ(x, state) → feature vector.

All 9 features are derived from SLSState's incremental tables and on-demand
break/make methods. Features are cached within a single call to avoid redundant
computation when multiple features share the same underlying value.
"""

import numpy as np
from src.sat.state import SLSState


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
    "interian":   ["break", "is_recent_5", "is_recent_10"],
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

    vec = []
    for name in names:
        if name == "break":
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
