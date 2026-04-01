"""
Feature extraction: φ(x, state) → feature vector.

All 9 features are O(1) lookups from SLSState's incremental tables.
Feature sets are defined here so ablations can be configured by name.
"""

import numpy as np
from src.sat.state import SLSState


# Ordered list of all available features.
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

# Named ablation sets (subsets of ALL_FEATURES).
FEATURE_SETS = {
    "interian": ["break", "is_recent_5", "is_recent_10"],
    "base":     ["make", "break", "age", "is_recent_5", "is_recent_10"],
    "full":     ALL_FEATURES,
    "no_recency": ["make", "break", "age", "break_zero", "unsat_deg", "deg", "flip_count"],
}


def extract(var: int, state: SLSState, feature_set: str = "full") -> np.ndarray:
    """
    Extract the feature vector for variable var in the given state.
    Returns shape (len(features),) float32 array.
    """
    names = FEATURE_SETS[feature_set]
    return extract_named(var, state, names)


def extract_named(var: int, state: SLSState, names: list[str]) -> np.ndarray:
    """Extract a specific list of features by name."""
    vec = []
    age = state.age(var)
    for name in names:
        if name == "break":
            vec.append(float(state.break_[var]))
        elif name == "make":
            vec.append(float(state.make[var]))
        elif name == "age":
            vec.append(float(age))
        elif name == "is_recent_5":
            vec.append(float(age <= 5))
        elif name == "is_recent_10":
            vec.append(float(age <= 10))
        elif name == "break_zero":
            vec.append(float(state.break_[var] == 0))
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
