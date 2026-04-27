"""
Evaluate random 3-SAT test performance for the paper's final comparators.

This script mirrors the in-domain evaluation flow but is restricted to:
  - random_3sat
  - scales n40 and n50
  - MinBreak, Novelty+, Interian, and LTF-KL (run7)

It reuses cached per-instance raw results when available, writes a summary CSV
with bootstrap confidence intervals, and produces family-level plots.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from plot_in_domain import (
    BUDGETS,
    MethodSpec,
    evaluate_method,
    has_cached_raw,
    load_formulas,
    plot_family,
    scale_to_int,
    setup_logging,
    summarize_rows,
    write_summary_csv,
)


RANDOM_3SAT_SCALES = ["n40", "n50"]


def available_random_methods(raw_dir: Path, split: str, scale: str) -> list[MethodSpec]:
    specs = [
        MethodSpec(label="MinBreak", kind="minbreak"),
        MethodSpec(label="Novelty+", kind="noveltyplus"),
    ]

    interian_ckpt = Path("experiments/models/random_3sat") / scale / "interian" / "best_interian.pt"
    if interian_ckpt.exists() or has_cached_raw(raw_dir, "random_3sat", scale, split, "Interian"):
        specs.append(MethodSpec(label="Interian", kind="interian", checkpoint=interian_ckpt))

    run7_ckpt = Path("experiments/models/random_3sat") / scale / "mlp_full_e_kl_sm" / "best_mlp_full_e_kl_sm.pt"
    if run7_ckpt.exists() or has_cached_raw(raw_dir, "random_3sat", scale, split, "LTF-KL"):
        specs.append(MethodSpec(label="LTF-KL", kind="mlp", checkpoint=run7_ckpt))

    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate random 3-SAT test performance for final paper methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-tries", type=int, default=10)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("experiments/results/random_3sat_test/raw"),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("experiments/plots/random_3sat_test"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("experiments/plots/random_3sat_test/random_3sat_summary.csv"),
    )
    parser.add_argument("--force", action="store_true", help="Recompute raw per-instance results even if cached.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_summary_rows: list[dict] = []
    family = "random_3sat"

    for scale in RANDOM_3SAT_SCALES:
        methods = available_random_methods(args.raw_dir, args.split, scale)
        if not methods:
            print(f"\n=== {family}/{scale} ({args.split}) ===")
            print("No available methods found; skipping.")
            continue

        formulas = load_formulas(family, scale, args.split, args.max_instances)
        if not formulas:
            print(f"\n=== {family}/{scale} ({args.split}) ===")
            print("No instances found; skipping.")
            continue

        print(f"\n=== {family}/{scale} ({args.split}) ===")
        print(f"Instances: {len(formulas)}")

        for spec in methods:
            print(f"  {spec.label:<10} ...", end="", flush=True)
            rows = evaluate_method(
                family=family,
                scale=scale,
                split=args.split,
                spec=spec,
                max_tries=args.max_tries,
                max_instances=args.max_instances,
                raw_dir=args.raw_dir,
                force=args.force,
            )
            if not rows:
                print(" skipped")
                continue
            summary = summarize_rows(
                rows=rows,
                max_flips=BUDGETS[scale],
                family=family,
                scale=scale,
                method=spec.label,
                n_boot=args.bootstrap_samples,
                seed=args.seed + scale_to_int(scale),
            )
            all_summary_rows.append(summary)
            print(
                f" median={summary['median_flips']:.1f}"
                f" mean={summary['mean_flips']:.1f}"
                f" par10={summary['par10']:.1f}"
                f" solve={summary['solve_rate']:.1f}%"
                f" time={summary['mean_time_s']:.4f}s"
            )

    if not all_summary_rows:
        raise RuntimeError("No random 3-SAT results found to summarize.")

    all_summary_rows.sort(key=lambda row: (row["family"], row["size"], row["method"]))
    write_summary_csv(args.summary_csv, all_summary_rows)
    plot_family(all_summary_rows, family, args.plot_dir)

    print(f"\nSummary CSV: {args.summary_csv}")
    print(f"Plots saved to: {args.plot_dir}")


if __name__ == "__main__":
    main()
