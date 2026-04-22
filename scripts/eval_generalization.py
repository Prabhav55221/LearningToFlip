"""
Zero-shot generalization evaluation.

Evaluates models trained on one scale across all scales of the same family.
Compares our method vs interian (both trained on the same source scale) vs baselines.

Metrics per instance: solved, n_flips, n_tries, time_s
Aggregate per (scale, policy): median_flips, mean_flips, solve_rate, par10,
                                mean_time_s, median_time_s, total_time_s

Budget scales with each test scale (same for all policies at that scale).

Usage:
    python scripts/eval_generalization.py \
        --family kcoloring \
        --train-scale n50 \
        --our-model experiments/models/kcoloring/n50/mlp_full_e/best_mlp_full_e.pt \
        --our-feature-set full \
        --interian-model experiments/models/kcoloring/n50/interian/best_interian.pt \
        --split test \
        --save-dir experiments/generalization/kcoloring_from_n50

Outputs:
    <save-dir>/summary.csv          one row per (scale, policy), all aggregate stats
    <save-dir>/<scale>_instances.csv  per-instance rows for all policies at that scale
"""

import argparse
import csv
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.utils.logging import setup as setup_logging
from src.sat.parser import parse_dimacs
from src.sls.solver import solve
from src.policy.baselines import MinBreak, NoveltyPlus
from src.policy.linear import LinearPolicy
from src.policy.mlp import MLPPolicy
from src.train.online import OnlineAdapter, OnlineConfig

log = logging.getLogger(__name__)

BUDGETS = {
    "n5":   500,   "n7":   700,   "n9":   900,
    "n10": 1_000,  "n12": 1_200,  "n15": 1_500,  "n20": 2_000,
    "n40": 4_000,  "n50": 5_000,  "n60": 6_000,  "n70": 7_000,
    "n75": 7_500,  "n100": 10_000, "n200": 20_000, "n300": 30_000,
}

FAMILY_SCALES = {
    "kcoloring":   ["n50", "n75", "n100", "n200", "n300"],
    "random_3sat": ["n40", "n50", "n60", "n70", "n100", "n200"],
    "kclique":     ["n5", "n10", "n15", "n20"],
    "domset":      ["n5", "n7", "n9", "n12"],
}


# ── Policy loading ────────────────────────────────────────────────────────────

def _load_mlp(path: Path, feature_set: str, normalize: bool = False) -> MLPPolicy:
    p = MLPPolicy(feature_set=feature_set, normalize=normalize)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    p.load_state_dict(sd)
    p.eval()
    return p


def load_policies(args) -> dict:
    """Returns ordered dict of {display_name: policy}."""
    policies = {}
    policies["minbreak"]    = MinBreak()
    policies["noveltyplus"] = NoveltyPlus()

    tag = f"n{args.train_scale[1:]}"

    if args.interian_model:
        p = LinearPolicy(feature_set="interian")
        sd = torch.load(args.interian_model, map_location="cpu", weights_only=True)
        p.load_state_dict(sd)
        p.eval()
        policies[f"interian({tag})"] = p

    if args.our_model:
        p = _load_mlp(args.our_model, args.our_feature_set)
        policies[f"ours({tag})"] = p
        if args.online:
            cfg     = OnlineConfig(k=args.online_k, lr=args.online_lr,
                                   entropy_coef=args.online_entropy_coef)
            policies[f"ours+online_k{args.online_k}({tag})"] = OnlineAdapter(p, cfg)

    if args.our_norm_model:
        p = _load_mlp(args.our_norm_model, args.our_feature_set, normalize=True)
        policies[f"ours_norm({tag})"] = p
        if args.online:
            cfg     = OnlineConfig(k=args.online_k, lr=args.online_lr,
                                   entropy_coef=args.online_entropy_coef)
            policies[f"ours_norm+online_k{args.online_k}({tag})"] = OnlineAdapter(p, cfg)

    if args.multiscale_model:
        ms_tag = args.multiscale_train_scale or "ms"
        p = _load_mlp(args.multiscale_model, args.our_feature_set)
        policies[f"ours_ms({ms_tag})"] = p
        if args.online:
            cfg     = OnlineConfig(k=args.online_k, lr=args.online_lr,
                                   entropy_coef=args.online_entropy_coef)
            policies[f"ours_ms+online_k{args.online_k}({ms_tag})"] = OnlineAdapter(p, cfg)

    if args.norm_multiscale_model:
        ms_tag = args.multiscale_train_scale or "ms"
        p = _load_mlp(args.norm_multiscale_model, args.our_feature_set, normalize=True)
        policies[f"ours_ms_norm({ms_tag})"] = p
        if args.online:
            cfg     = OnlineConfig(k=args.online_k, lr=args.online_lr,
                                   entropy_coef=args.online_entropy_coef)
            policies[f"ours_ms_norm+online_k{args.online_k}({ms_tag})"] = OnlineAdapter(p, cfg)

    return policies


# ── Per-instance evaluation ───────────────────────────────────────────────────

def eval_policy_on_scale(policy, formulas, max_flips, max_tries):
    """Run policy on all formulas; return list of per-instance metric dicts.

    Handles both standard policies (via sls.solver.solve) and OnlineAdapter
    (via its own .solve() which interleaves gradient updates).
    """
    is_online = isinstance(policy, OnlineAdapter)
    records = []
    for i, formula in enumerate(formulas):
        t0 = time.perf_counter()
        if is_online:
            result = policy.solve(formula, max_flips, max_tries, reset=True)
        else:
            with torch.no_grad():
                result = solve(formula, policy, max_flips, max_tries)
        elapsed = time.perf_counter() - t0
        records.append({
            "instance_idx": i,
            "solved":       result.solved,
            "n_flips":      result.n_flips,
            "n_tries":      result.n_tries,
            "time_s":       round(elapsed, 4),
        })
    return records


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(records, max_flips):
    n        = len(records)
    n_solved = sum(1 for r in records if r["solved"])
    flips    = [r["n_flips"] for r in records]
    times    = [r["time_s"]  for r in records]
    par10    = float(np.mean([
        f if r["solved"] else 10 * max_flips
        for f, r in zip(flips, records)
    ]))
    return {
        "n_instances":   n,
        "n_solved":      n_solved,
        "solve_rate":    round(n_solved / n * 100, 1),
        "median_flips":  round(float(np.median(flips)), 1),
        "mean_flips":    round(float(np.mean(flips)),   1),
        "par10":         round(par10, 1),
        "median_time_s": round(float(np.median(times)), 4),
        "mean_time_s":   round(float(np.mean(times)),   4),
        "total_time_s":  round(float(np.sum(times)),    2),
    }


# ── Pretty printing ───────────────────────────────────────────────────────────

def print_scale_table(scale, max_flips, results):
    print(f"\n{'='*80}")
    print(f"  {scale}  (budget={max_flips:,} flips/try)")
    print(f"{'='*80}")
    hdr = (f"{'Policy':<22}  {'Median':>7}  {'Mean':>8}  {'PAR-10':>9}  "
           f"{'Solve%':>7}  {'Solved':>8}  {'AvgTime':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, agg in results.items():
        print(
            f"{name:<22}  "
            f"{agg['median_flips']:>7.0f}  "
            f"{agg['mean_flips']:>8.0f}  "
            f"{agg['par10']:>9.0f}  "
            f"{agg['solve_rate']:>6.1f}%  "
            f"{agg['n_solved']:>4}/{agg['n_instances']:<3}  "
            f"{agg['mean_time_s']:>7.3f}s"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot generalization evaluation across scales",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--family",        choices=list(FAMILY_SCALES.keys()), required=True)
    parser.add_argument("--train-scale",   choices=list(BUDGETS.keys()),       required=True,
                        help="Scale the models were trained on.")
    parser.add_argument("--test-scales",   nargs="+", default=None,
                        help="Scales to test on. Defaults to all scales for the family.")
    parser.add_argument("--our-model",     type=Path, default=None,
                        help="Path to our trained MLP checkpoint (base, trained on --train-scale).")
    parser.add_argument("--our-feature-set", default="full",
                        help="Feature set used by our MLP models.")
    parser.add_argument("--our-norm-model", type=Path, default=None,
                        help="Path to our MLP trained with --normalize-features on --train-scale.")
    parser.add_argument("--multiscale-model", type=Path, default=None,
                        help="Path to our MLP trained on multiple scales (without normalization).")
    parser.add_argument("--norm-multiscale-model", type=Path, default=None,
                        help="Path to our MLP trained on multiple scales with normalization.")
    parser.add_argument("--multiscale-train-scale", default=None,
                        help="Label for the multi-scale training set, e.g. 'n5-n10-n15' (display only).")
    parser.add_argument("--interian-model", type=Path, default=None,
                        help="Path to Interian checkpoint trained on --train-scale.")
    parser.add_argument("--split",         choices=["val", "test"], default="test")
    parser.add_argument("--max-tries",     type=int, default=10)
    parser.add_argument("--max-instances", type=int, default=None,
                        help="Cap instances per scale (quick dev runs).")
    parser.add_argument("--save-dir",      type=Path,
                        default=Path("experiments/generalization"))
    parser.add_argument("--online",        action="store_true",
                        help="Also evaluate online adaptation for all loaded MLP models.")
    parser.add_argument("--online-k",      type=int, default=10,
                        help="k-step horizon for online adaptation buffer.")
    parser.add_argument("--online-lr",     type=float, default=1e-5,
                        help="Learning rate for online adaptation.")
    parser.add_argument("--online-entropy-coef", type=float, default=0.01,
                        help="Entropy coefficient for online adaptation.")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--verbose",       action="store_true")
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
    instance_fieldnames = ["scale", "policy", "instance_idx", "solved",
                           "n_flips", "n_tries", "time_s"]
    summary_fieldnames  = ["scale", "policy", "train_scale", "n_instances",
                           "n_solved", "solve_rate", "median_flips", "mean_flips",
                           "par10", "median_time_s", "mean_time_s", "total_time_s"]

    for scale in test_scales:
        data_dir = Path("data") / args.family / scale / args.split
        if not data_dir.exists():
            log.warning("No data at %s — skipping", data_dir)
            continue

        paths = sorted(data_dir.glob("*.cnf"))
        if not paths:
            log.warning("No .cnf files in %s — skipping", data_dir)
            continue
        if args.max_instances:
            paths = paths[:args.max_instances]

        max_flips = BUDGETS[scale]
        print(f"\nLoading {len(paths)} instances from {data_dir} (budget={max_flips:,}) ...")
        formulas = [parse_dimacs(p) for p in paths]

        scale_results  = {}   # {policy_name: agg_dict}
        instance_rows  = []   # per-instance rows for this scale

        for name, policy in policies.items():
            print(f"  {name} ...", end="", flush=True)
            t_scale = time.perf_counter()
            records = eval_policy_on_scale(policy, formulas, max_flips, args.max_tries)
            agg     = aggregate(records, max_flips)
            scale_results[name] = agg
            print(f"  median={agg['median_flips']:.0f}  solve={agg['solve_rate']}%  "
                  f"par10={agg['par10']:.0f}  ({time.perf_counter()-t_scale:.1f}s)")

            for r in records:
                instance_rows.append({
                    "scale":        scale,
                    "policy":       name,
                    **r,
                })
            summary_rows.append({
                "scale":       scale,
                "policy":      name,
                "train_scale": args.train_scale,
                **agg,
            })

        print_scale_table(scale, max_flips, scale_results)

        # Save per-scale instance CSV
        inst_path = args.save_dir / f"{scale}_instances.csv"
        with open(inst_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=instance_fieldnames)
            w.writeheader()
            w.writerows(instance_rows)

    # Save summary CSV
    summary_path = args.save_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nSummary saved to {summary_path}")
    print(f"Per-instance CSVs saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
