"""
Unified evaluation script.

Compare classical baselines and trained learned policies on any data split.
Metrics match Interian & Bernardini (KR 2023): median flips, solve rate, mean flips.
Unsolved instances count as max_flips in all aggregates.

Usage:
    # Classical baselines on val (default)
    python scripts/evaluate.py --family kcoloring --scale n100

    # Add trained Interian policy
    python scripts/evaluate.py --family kcoloring --scale n100 \\
        --policies minbreak noveltyplus interian \\
        --model-path experiments/models/kcoloring/n100/interian/best_interian.pt

    # Quick smoke test (5 instances)
    python scripts/evaluate.py --family kcoloring --scale n100 --max-instances 5

    # Final evaluation — keep test locked during development
    python scripts/evaluate.py --family kcoloring --scale n100 --split test \\
        --policies minbreak noveltyplus interian \\
        --model-path experiments/models/kcoloring/n100/interian/best_interian.pt
"""

import argparse
import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch

from src.utils.logging import setup as setup_logging
from src.sat.parser import parse_dimacs
from src.sls.solver import solve
from src.policy.baselines import MinBreak, NoveltyPlus
from src.policy.linear import LinearPolicy
from src.policy.mlp import MLPPolicy


BUDGETS = {"n100": 10_000, "n200": 50_000}

log = logging.getLogger(__name__)


def load_policy(name: str, model_path: Path | None):
    """Instantiate and optionally load weights for a named policy."""
    if name == "minbreak":
        return MinBreak()
    if name == "noveltyplus":
        return NoveltyPlus()
    if name == "interian":
        if model_path is None:
            raise ValueError("--model-path is required for 'interian' policy")
        policy = LinearPolicy(feature_set="interian")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        policy.load_state_dict(state_dict)
        policy.eval()
        return policy
    if name == "mlp":
        if model_path is None:
            raise ValueError("--model-path is required for 'mlp' policy")
        policy = MLPPolicy()
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        policy.load_state_dict(state_dict)
        policy.eval()
        return policy
    raise ValueError(f"Unknown policy: {name!r}")


def evaluate_policy(policy, formulas: list, max_flips: int, max_tries: int) -> dict:
    """Run policy on all formulas. Returns aggregated stats."""
    flip_counts = []
    for formula in formulas:
        with torch.no_grad():
            result = solve(formula, policy, max_flips, max_tries)
        flip_counts.append(result.n_flips)

    n = len(flip_counts)
    n_solved = sum(1 for f in flip_counts if f < max_flips)
    return {
        "median_flips": float(np.median(flip_counts)),
        "mean_flips":   float(np.mean(flip_counts)),
        "solve_rate":   n_solved / n * 100.0,
        "n_instances":  n,
        "n_solved":     n_solved,
    }


def print_table(results: dict[str, dict], max_flips: int) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        f"{'Policy':<14}  {'Median Flips':>12}  {'Solve Rate':>10}  "
        f"{'Mean Flips':>10}  {'Solved':>8}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for name, stats in results.items():
        print(
            f"{name:<14}  "
            f"{stats['median_flips']:>12.0f}  "
            f"{stats['solve_rate']:>9.1f}%  "
            f"{stats['mean_flips']:>10.0f}  "
            f"{stats['n_solved']:>4}/{stats['n_instances']:<3}"
        )
    print(sep)
    print(f"  budget: {max_flips:,} flips per try · unsolved → counted as {max_flips:,}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SLS policies on SAT instances",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--family",   choices=["kcoloring", "random_3sat"], required=True)
    parser.add_argument("--scale",    choices=["n100", "n200"],             required=True)
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="val",
        help="Data split to evaluate on. Keep 'test' locked until final reporting.",
    )
    parser.add_argument(
        "--policies", nargs="+",
        choices=["minbreak", "noveltyplus", "interian", "mlp"],
        default=["minbreak", "noveltyplus"],
        help="Policies to evaluate.",
    )
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="Path to trained model checkpoint (required for interian/mlp).",
    )
    parser.add_argument(
        "--max-tries", type=int, default=10,
        help="Number of random restarts per instance.",
    )
    parser.add_argument(
        "--max-instances", type=int, default=None,
        help="Evaluate on first N instances only (quick dev runs).",
    )
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--verbose",  action="store_true", help="Log per-policy progress.")
    parser.add_argument(
        "--save-csv", type=Path, default=None,
        help="Save results table to this CSV file.",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    max_flips = BUDGETS[args.scale]

    # Load instances
    data_dir = Path("data") / args.family / args.scale / args.split
    paths = sorted(data_dir.glob("*.cnf"))
    if not paths:
        raise FileNotFoundError(f"No .cnf files found in {data_dir}")
    if args.max_instances is not None:
        paths = paths[: args.max_instances]

    print(f"Loading {len(paths)} instances from {data_dir} ...")
    formulas = [parse_dimacs(p) for p in paths]

    # Evaluate each policy in order
    all_results: dict[str, dict] = {}
    for policy_name in args.policies:
        print(f"  Running {policy_name} ...", end="", flush=True)
        policy = load_policy(policy_name, args.model_path)
        stats = evaluate_policy(policy, formulas, max_flips, args.max_tries)
        all_results[policy_name] = stats
        print(
            f"  median={stats['median_flips']:.0f}  "
            f"solve={stats['solve_rate']:.1f}%"
        )

    # Print comparison table
    print(f"\n=== {args.family}/{args.scale} — {args.split} split ===")
    print_table(all_results, max_flips)

    # Optionally save CSV
    if args.save_csv is not None:
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["policy", "median_flips", "mean_flips",
                            "solve_rate", "n_solved", "n_instances"],
            )
            writer.writeheader()
            for name, stats in all_results.items():
                writer.writerow({"policy": name, **stats})
        print(f"Results saved to {args.save_csv}")


if __name__ == "__main__":
    main()
