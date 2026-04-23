"""
In-domain test-set evaluation and plotting.

This script evaluates all available in-domain checkpoints on the matching test
split, saves per-instance raw results, computes paper-friendly summary metrics
with bootstrap confidence intervals, and generates family-level line plots.

Methods:
  - MinBreak
  - Novelty+
  - Interian
  - LTF-Naive   (run6e / mlp_full_e)
  - LTF-KL      (run7 / mlp_full_e_kl_sm)

Missing checkpoints are skipped. Baselines are always run on the fly.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.policy.baselines import MinBreak, NoveltyPlus
from src.policy.linear import LinearPolicy
from src.policy.mlp import MLPPolicy
from src.sat.parser import parse_dimacs
from src.sls.solver import solve
from src.utils.logging import setup as setup_logging


log = logging.getLogger(__name__)

BUDGETS = {
    "n5": 500,
    "n7": 700,
    "n9": 900,
    "n10": 1_000,
    "n12": 1_200,
    "n15": 1_500,
    "n20": 2_000,
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
    "kcoloring": ["n50", "n75", "n100", "n200", "n300"],
    "random_3sat": ["n40", "n50", "n60", "n70", "n100", "n200"],
    "kclique": ["n5", "n10", "n15", "n20"],
    "domset": ["n5", "n7", "n9", "n12"],
}

FAMILY_TITLES = {
    "domset": "Dominating Set",
    "kclique": "k-Clique",
    "kcoloring": "k-Coloring",
    "random_3sat": "Random 3-SAT",
}

METHOD_STYLES = {
    "MinBreak": {"color": "#4C78A8", "marker": "o"},
    "Novelty+": {"color": "#F58518", "marker": "s"},
    "Interian": {"color": "#54A24B", "marker": "^"},
    "LTF-Naive": {"color": "#E45756", "marker": "D"},
    "LTF-KL": {"color": "#B279A2", "marker": "P"},
}


@dataclass(frozen=True)
class MethodSpec:
    label: str
    kind: str
    checkpoint: Path | None = None


def scale_to_int(scale: str) -> int:
    return int(scale[1:])


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


def load_policy(spec: MethodSpec):
    if spec.kind == "minbreak":
        return MinBreak()
    if spec.kind == "noveltyplus":
        return NoveltyPlus()
    if spec.kind == "interian":
        assert spec.checkpoint is not None
        policy = LinearPolicy(feature_set="interian")
        policy.load_state_dict(torch.load(spec.checkpoint, map_location="cpu", weights_only=True))
        policy.eval()
        return policy
    if spec.kind == "mlp":
        assert spec.checkpoint is not None
        hidden_dim, n_layers = infer_mlp_architecture(spec.checkpoint)
        policy = MLPPolicy(
            feature_set="full",
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            normalize=False,
        )
        policy.load_state_dict(torch.load(spec.checkpoint, map_location="cpu", weights_only=True))
        policy.eval()
        return policy
    raise ValueError(f"Unknown method kind: {spec.kind}")


def available_methods(family: str, scale: str) -> list[MethodSpec]:
    specs = [
        MethodSpec(label="MinBreak", kind="minbreak"),
        MethodSpec(label="Novelty+", kind="noveltyplus"),
    ]

    interian_ckpt = Path("experiments/models") / family / scale / "interian" / "best_interian.pt"
    if interian_ckpt.exists():
        specs.append(MethodSpec(label="Interian", kind="interian", checkpoint=interian_ckpt))

    run6e_ckpt = Path("experiments/models") / family / scale / "mlp_full_e" / "best_mlp_full_e.pt"
    if run6e_ckpt.exists():
        specs.append(MethodSpec(label="LTF-Naive", kind="mlp", checkpoint=run6e_ckpt))

    run7_ckpt = Path("experiments/models") / family / scale / "mlp_full_e_kl_sm" / "best_mlp_full_e_kl_sm.pt"
    if run7_ckpt.exists():
        specs.append(MethodSpec(label="LTF-KL", kind="mlp", checkpoint=run7_ckpt))

    # Skip scales that only have classical baselines; the user asked for
    # "whatever we have", meaning currently available learned runs.
    learned_labels = {"Interian", "LTF-Naive", "LTF-KL"}
    if not any(spec.label in learned_labels for spec in specs):
        return []
    return specs


def load_formulas(family: str, scale: str, split: str, max_instances: int | None):
    data_dir = Path("data") / family / scale / split
    if not data_dir.exists():
        return []
    paths = sorted(data_dir.glob("*.cnf"))
    if max_instances is not None:
        paths = paths[:max_instances]
    return [(path.name, parse_dimacs(path)) for path in paths]


def raw_csv_path(raw_dir: Path, family: str, scale: str, split: str, method_label: str) -> Path:
    slug = method_label.lower().replace("+", "plus").replace("-", "_").replace(" ", "_")
    return raw_dir / family / f"{scale}_{split}_{slug}.csv"


def evaluate_method(
    family: str,
    scale: str,
    split: str,
    spec: MethodSpec,
    max_tries: int,
    max_instances: int | None,
    raw_dir: Path,
    force: bool,
) -> list[dict]:
    out_path = raw_csv_path(raw_dir, family, scale, split, spec.label)
    if out_path.exists() and not force:
        with out_path.open() as f:
            return [
                {
                    "instance": row["instance"],
                    "solved": row["solved"] == "True",
                    "n_flips": float(row["n_flips"]),
                    "n_tries": float(row["n_tries"]),
                    "time_s": float(row["time_s"]),
                }
                for row in csv.DictReader(f)
            ]

    formulas = load_formulas(family, scale, split, max_instances)
    if not formulas:
        return []

    policy = load_policy(spec)
    max_flips = BUDGETS[scale]
    rows = []

    for instance_name, formula in formulas:
        start = time.perf_counter()
        with torch.no_grad():
            result = solve(formula, policy, max_flips, max_tries)
        elapsed = time.perf_counter() - start
        rows.append({
            "instance": instance_name,
            "solved": bool(result.solved),
            "n_flips": float(result.n_flips),
            "n_tries": float(result.n_tries),
            "time_s": float(elapsed),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["instance", "solved", "n_flips", "n_tries", "time_s"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def bootstrap_ci(
    n_items: int,
    stat_fn,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if n_items == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        stats[i] = stat_fn(idx)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_rows(
    rows: list[dict],
    max_flips: int,
    family: str,
    scale: str,
    method: str,
    n_boot: int,
    seed: int,
) -> dict:
    flips = np.array([row["n_flips"] for row in rows], dtype=float)
    solved = np.array([row["solved"] for row in rows], dtype=bool)
    times = np.array([row["time_s"] for row in rows], dtype=float)
    tries = np.array([row["n_tries"] for row in rows], dtype=float)
    par10_values = np.where(solved, flips, 10.0 * max_flips)
    solved_flips = flips[solved]

    def on_sample(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return values[idx]

    summary = {
        "family": family,
        "scale": scale,
        "size": scale_to_int(scale),
        "method": method,
        "n_instances": int(len(rows)),
        "n_solved": int(np.sum(solved)),
        "solve_rate": float(np.mean(solved) * 100.0),
        "median_flips": float(np.median(flips)),
        "mean_flips": float(np.mean(flips)),
        "par10": float(np.mean(par10_values)),
        "median_time_s": float(np.median(times)),
        "mean_time_s": float(np.mean(times)),
        "total_time_s": float(np.sum(times)),
        "median_tries": float(np.median(tries)),
        "mean_tries": float(np.mean(tries)),
        "std_flips": float(np.std(flips)),
        "iqr_flips": float(np.percentile(flips, 75) - np.percentile(flips, 25)),
        "mean_flips_solved_only": float(np.mean(solved_flips)) if solved_flips.size else float("nan"),
        "median_flips_solved_only": float(np.median(solved_flips)) if solved_flips.size else float("nan"),
    }

    ci_stats = {
        "solve_rate": lambda idx: np.mean(on_sample(solved.astype(float) * 100.0, idx)),
        "median_flips": lambda idx: np.median(on_sample(flips, idx)),
        "mean_flips": lambda idx: np.mean(on_sample(flips, idx)),
        "par10": lambda idx: np.mean(on_sample(par10_values, idx)),
        "median_time_s": lambda idx: np.median(on_sample(times, idx)),
        "mean_time_s": lambda idx: np.mean(on_sample(times, idx)),
        "mean_tries": lambda idx: np.mean(on_sample(tries, idx)),
    }

    for offset, (name, stat_fn) in enumerate(ci_stats.items()):
        lo, hi = bootstrap_ci(len(rows), stat_fn, n_boot=n_boot, seed=seed + offset)
        summary[f"{name}_ci_low"] = lo
        summary[f"{name}_ci_high"] = hi

    return summary


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_family(summary_rows: list[dict], family: str, out_dir: Path) -> None:
    family_rows = [row for row in summary_rows if row["family"] == family]
    if not family_rows:
        return

    metrics = [
        ("median_flips", "Median Flips"),
        ("mean_flips", "Mean Flips"),
        ("par10", "PAR-10"),
        ("solve_rate", "Solve Rate (%)"),
        ("mean_time_s", "Mean Time (s)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes_flat = axes.flatten()

    for ax_idx, (metric, title) in enumerate(metrics):
        ax = axes_flat[ax_idx]
        for method, style in METHOD_STYLES.items():
            method_rows = sorted(
                (row for row in family_rows if row["method"] == method),
                key=lambda row: row["size"],
            )
            if not method_rows:
                continue
            x = np.array([row["size"] for row in method_rows], dtype=float)
            y = np.array([row[metric] for row in method_rows], dtype=float)
            lo_key = f"{metric}_ci_low"
            hi_key = f"{metric}_ci_high"
            has_ci = all(lo_key in row and hi_key in row for row in method_rows)
            ax.plot(
                x,
                y,
                label=method,
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=6.0,
            )
            if has_ci:
                lo = np.array([row[lo_key] for row in method_rows], dtype=float)
                hi = np.array([row[hi_key] for row in method_rows], dtype=float)
                ax.fill_between(x, lo, hi, color=style["color"], alpha=0.15)

        ax.set_title(title)
        ax.set_xlabel("Size")
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
        if metric == "solve_rate":
            ax.set_ylim(0.0, 102.0)

    axes_flat[5].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        axes_flat[5].legend(handles, labels, loc="center", frameon=False, ncol=1)

    fig.suptitle(f"{FAMILY_TITLES.get(family, family)} In-Domain Test Performance", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{family}_in_domain_test.png"
    pdf_path = out_dir / f"{family}_in_domain_test.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and plot in-domain test performance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--families", nargs="+", choices=list(FAMILY_SCALES.keys()), default=list(FAMILY_SCALES.keys()))
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max-tries", type=int, default=10)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-dir", type=Path, default=Path("experiments/results/in_domain_test/raw"))
    parser.add_argument("--plot-dir", type=Path, default=Path("experiments/plots/in_domain_test"))
    parser.add_argument("--summary-csv", type=Path, default=Path("experiments/plots/in_domain_test/in_domain_summary.csv"))
    parser.add_argument("--force", action="store_true", help="Recompute raw per-instance results even if cached.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_summary_rows = []

    for family in args.families:
        for scale in FAMILY_SCALES[family]:
            methods = available_methods(family, scale)
            if not methods:
                log.info("Skipping %s/%s: no learned checkpoints available", family, scale)
                continue

            formulas = load_formulas(family, scale, args.split, args.max_instances)
            if not formulas:
                log.info("Skipping %s/%s: no %s instances found", family, scale, args.split)
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
        raise RuntimeError("No in-domain results found to summarize.")

    all_summary_rows.sort(key=lambda row: (row["family"], row["size"], row["method"]))
    write_summary_csv(args.summary_csv, all_summary_rows)

    for family in args.families:
        plot_family(all_summary_rows, family, args.plot_dir)

    print(f"\nSummary CSV: {args.summary_csv}")
    print(f"Plots saved to: {args.plot_dir}")


if __name__ == "__main__":
    main()
