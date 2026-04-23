"""
Generalization evaluation for one learned checkpoint across the family scales.

The workflow assumes a single base checkpoint at a time, currently run6e by
default, and compares:
  - classical baselines
  - optional Interian checkpoint
  - frozen learned policy
  - two online adaptation methods
"""

import argparse
import csv
import logging
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from src.policy.baselines import MinBreak, NoveltyPlus
from src.policy.linear import LinearPolicy
from src.policy.mlp import MLPPolicy
from src.sat.parser import parse_dimacs
from src.sls.solver import run_try
from src.train.online import (
    OnlineKLAdapter,
    OnlineKLConfig,
    OnlineSuccessKLAdapter,
    OnlineSuccessKLConfig,
)
from src.utils.logging import setup as setup_logging

log = logging.getLogger(__name__)

BUDGETS = {
    "n3": 300,
    "n5": 500,
    "n7": 700,
    "n9": 900,
    "n10": 1_000,
    "n12": 1_200,
    "n15": 1_500,
    "n20": 2_000,
    "n30": 3_000,
    "n40": 4_000,
    "n50": 5_000,
    "n60": 6_000,
    "n70": 7_000,
    "n75": 7_500,
    "n100": 10_000,
    "n200": 20_000,
    "n300": 30_000,
}

FAMILY_SCALES = {
    "kcoloring": ["n10", "n20", "n30", "n50", "n75", "n100"],
    "random_3sat": ["n40", "n50", "n60", "n70", "n100", "n200"],
    "kclique": ["n3", "n5", "n10", "n12", "n15", "n20"],
    "domset": ["n5", "n7", "n9", "n12"],
}

METHOD_CHOICES = [
    "minbreak",
    "noveltyplus",
    "interian",
    "ours_frozen",
    "ours_online_kl",
    "ours_online_success_kl",
]


def infer_mlp_architecture(model_path: Path) -> tuple[int, int]:
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    linear_keys = sorted(
        (
            key for key in state_dict
            if re.fullmatch(r"net\.\d+\.weight", key)
        ),
        key=lambda key: int(key.split(".")[1]),
    )
    if not linear_keys:
        raise ValueError(f"Could not infer MLP architecture from {model_path}")

    hidden_dim = int(state_dict[linear_keys[0]].shape[0])
    n_layers = len(linear_keys) - 1
    return hidden_dim, n_layers


def load_mlp_policy(args) -> MLPPolicy:
    hidden_dim, n_layers = infer_mlp_architecture(args.model_path)
    policy = MLPPolicy(
        feature_set=args.feature_set,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        normalize=args.normalize_features,
    )
    state_dict = torch.load(args.model_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def load_interian_policy(model_path: Path) -> LinearPolicy:
    policy = LinearPolicy(feature_set="interian")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def load_policies(args) -> dict[str, object]:
    policies: dict[str, object] = {}
    base_label = f"ours({args.model_label})"
    online_label = f"ours_online_kl({args.model_label})"
    success_label = f"ours_online_success_kl({args.model_label})"
    interian_label = f"interian({args.train_scale})"

    for method in args.methods:
        if method == "minbreak":
            policies["minbreak"] = MinBreak()
        elif method == "noveltyplus":
            policies["noveltyplus"] = NoveltyPlus()
        elif method == "interian":
            if args.interian_model is None:
                raise ValueError("--interian-model is required when methods include 'interian'")
            policies[interian_label] = load_interian_policy(args.interian_model)
        elif method == "ours_frozen":
            policies[base_label] = load_mlp_policy(args)
        elif method == "ours_online_kl":
            policies[online_label] = OnlineKLAdapter(
                load_mlp_policy(args),
                OnlineKLConfig(
                    k=args.online_k,
                    gamma=args.gamma,
                    lr=args.online_lr,
                    entropy_coef=args.online_entropy_coef,
                    kl_anchor_coef=args.online_kl_anchor_coef,
                ),
            )
        elif method == "ours_online_success_kl":
            policies[success_label] = OnlineSuccessKLAdapter(
                load_mlp_policy(args),
                OnlineSuccessKLConfig(
                    gamma=args.gamma,
                    lr=args.online_success_lr,
                    kl_anchor_coef=args.online_success_kl_anchor_coef,
                ),
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    return policies


def eval_restart_policy(policy, formula, max_flips, max_tries):
    solved_any = False
    best_flips = max_flips
    cumulative_flips = 0

    with torch.no_grad():
        for _ in range(max_tries):
            result = run_try(formula, policy, max_flips)
            cumulative_flips += result.n_flips
            if result.solved and ((not solved_any) or (result.n_flips < best_flips)):
                solved_any = True
                best_flips = result.n_flips

    return {
        "solved": solved_any,
        "best_flips": best_flips,
        "cumulative_flips": cumulative_flips,
        "n_tries": max_tries,
    }


def eval_policy_on_scale(policy, formulas, max_flips, max_tries):
    is_online = isinstance(policy, (OnlineKLAdapter, OnlineSuccessKLAdapter))
    records = []

    for i, formula in enumerate(formulas):
        t0 = time.perf_counter()
        if is_online:
            result = policy.evaluate(formula, max_flips, max_tries, reset=True)
        else:
            result = eval_restart_policy(policy, formula, max_flips, max_tries)
        elapsed = time.perf_counter() - t0
        records.append({
            "instance_idx": i,
            "solved": result.solved if is_online else result["solved"],
            "best_flips": result.best_flips if is_online else result["best_flips"],
            "cumulative_flips": result.cumulative_flips if is_online else result["cumulative_flips"],
            "n_tries": result.n_tries if is_online else result["n_tries"],
            "time_s": round(elapsed, 4),
        })

    return records


def aggregate(records, max_flips, max_tries):
    n = len(records)
    n_solved = sum(1 for r in records if r["solved"])
    cumulative_flips = [r["cumulative_flips"] for r in records]
    best_flips = [r["best_flips"] for r in records]
    best_flips_solved = [r["best_flips"] for r in records if r["solved"]]
    times = [r["time_s"] for r in records]
    par10 = float(np.mean([
        f if r["solved"] else 10 * max_flips * max_tries
        for f, r in zip(cumulative_flips, records)
    ]))
    return {
        "n_instances": n,
        "n_solved": n_solved,
        "solve_rate": round(n_solved / n * 100, 1),
        "median_cumulative_flips": round(float(np.median(cumulative_flips)), 1),
        "mean_cumulative_flips": round(float(np.mean(cumulative_flips)), 1),
        # Conditional on solved instances only — avoids sentinel inflation at low solve rates.
        # None when nothing is solved.
        "median_best_flips_solved": round(float(np.median(best_flips_solved)), 1) if best_flips_solved else None,
        "mean_best_flips_solved": round(float(np.mean(best_flips_solved)), 1) if best_flips_solved else None,
        "par10": round(par10, 1),
        "median_time_s": round(float(np.median(times)), 4),
        "mean_time_s": round(float(np.mean(times)), 4),
        "total_time_s": round(float(np.sum(times)), 2),
    }


def print_scale_table(scale, max_flips, results):
    print(f"\n{'=' * 88}")
    print(f"  {scale}  (budget={max_flips:,} flips/try)")
    print(f"{'=' * 88}")
    header = (
        f"{'Policy':<30}  {'MedCum':>8}  {'MeanCum':>8}  {'MedBest|sol':>11}  "
        f"{'PAR-10':>9}  {'Solve%':>7}  {'Solved':>8}  {'AvgTime':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, agg in results.items():
        med_best = agg["median_best_flips_solved"]
        med_best_str = f"{med_best:>11.0f}" if med_best is not None else f"{'—':>11}"
        print(
            f"{name:<30}  "
            f"{agg['median_cumulative_flips']:>8.0f}  "
            f"{agg['mean_cumulative_flips']:>8.0f}  "
            f"{med_best_str}  "
            f"{agg['par10']:>9.0f}  "
            f"{agg['solve_rate']:>6.1f}%  "
            f"{agg['n_solved']:>4}/{agg['n_instances']:<3}  "
            f"{agg['mean_time_s']:>7.3f}s"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generalization evaluation for a single learned checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--family", choices=list(FAMILY_SCALES.keys()), required=True)
    parser.add_argument("--train-scale", choices=list(BUDGETS.keys()), required=True)
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Base learned checkpoint, currently assumed to be run6e by default.")
    parser.add_argument("--model-label", default="run6e",
                        help="Label used in output tables for the learned checkpoint.")
    parser.add_argument("--feature-set", default="full")
    parser.add_argument("--normalize-features", action="store_true")
    parser.add_argument("--interian-model", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", choices=METHOD_CHOICES,
                        default=["minbreak", "noveltyplus", "ours_frozen",
                                 "ours_online_kl", "ours_online_success_kl"])
    parser.add_argument("--test-scales", nargs="+", default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-tries", type=int, default=10)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--save-dir", type=Path, default=Path("experiments/generalization"))
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--online-k", type=int, default=10)
    parser.add_argument("--online-lr", type=float, default=1e-5)
    parser.add_argument("--online-entropy-coef", type=float, default=0.01)
    parser.add_argument("--online-kl-anchor-coef", type=float, default=0.05)
    parser.add_argument("--online-success-lr", type=float, default=1e-5)
    parser.add_argument("--online-success-kl-anchor-coef", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    policies = load_policies(args)
    print(f"\nPolicies: {list(policies.keys())}")

    test_scales = args.test_scales or FAMILY_SCALES[args.family]
    args.save_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    instance_fieldnames = [
        "scale", "policy", "instance_idx", "solved",
        "cumulative_flips", "best_flips", "n_tries", "time_s",
    ]
    summary_fieldnames = [
        "scale", "policy", "train_scale", "model_label", "n_instances", "n_solved",
        "solve_rate", "median_cumulative_flips", "mean_cumulative_flips",
        "median_best_flips_solved", "mean_best_flips_solved", "par10",
        "median_time_s", "mean_time_s", "total_time_s",
    ]

    for scale in test_scales:
        data_dir = Path("data") / args.family / scale / args.split
        if not data_dir.exists():
            log.warning("No data at %s — skipping", data_dir)
            continue

        paths = sorted(data_dir.glob("*.cnf"))
        if not paths:
            log.warning("No .cnf files in %s — skipping", data_dir)
            continue
        if args.max_instances is not None:
            paths = paths[:args.max_instances]

        max_flips = BUDGETS[scale]
        print(f"\nLoading {len(paths)} instances from {data_dir} (budget={max_flips:,}) ...")
        formulas = [parse_dimacs(p) for p in paths]

        scale_results = {}
        instance_rows = []

        for name, policy in policies.items():
            print(f"  {name} ...", end="", flush=True)
            t_scale = time.perf_counter()
            records = eval_policy_on_scale(policy, formulas, max_flips, args.max_tries)
            agg = aggregate(records, max_flips, args.max_tries)
            scale_results[name] = agg
            print(
                f"  median_cum={agg['median_cumulative_flips']:.0f}  solve={agg['solve_rate']}%  "
                f"par10={agg['par10']:.0f}  ({time.perf_counter() - t_scale:.1f}s)"
            )

            for record in records:
                instance_rows.append({
                    "scale": scale,
                    "policy": name,
                    **record,
                })

            summary_rows.append({
                "scale": scale,
                "policy": name,
                "train_scale": args.train_scale,
                "model_label": args.model_label,
                **agg,
            })

        print_scale_table(scale, max_flips, scale_results)

        inst_path = args.save_dir / f"{scale}_instances.csv"
        with open(inst_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=instance_fieldnames)
            writer.writeheader()
            writer.writerows(instance_rows)

    summary_path = args.save_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary saved to {summary_path}")
    print(f"Per-instance CSVs saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
