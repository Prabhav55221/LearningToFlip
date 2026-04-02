"""
Baseline evaluation script.

Usage:
    python test_run.py                          # summary only
    python test_run.py -v                       # per-instance results
    python test_run.py --debug                  # per-try detail
    python test_run.py -v --save-logs           # also write to logs/
    python test_run.py --family kcoloring       # single family
    python test_run.py --scale n100             # single scale
    python test_run.py --split val              # different data split
"""

import argparse
import logging
from pathlib import Path

from src.utils.logging import setup as setup_logging
from src.sat.parser import parse_dimacs
from src.policy.baselines import MinBreak, NoveltyPlus
from src.sls.solver import solve
from src.eval.metrics import summarise

log = logging.getLogger(__name__)

POLICIES = [
    ("MinBreak", MinBreak()),
    ("Novelty+", NoveltyPlus(p=0.1)),
]

BUDGETS = {"n100": (10_000, 10), "n200": (50_000, 10)}


def run(args: argparse.Namespace) -> None:
    families = [args.family] if args.family else ["kcoloring", "random_3sat"]
    scales   = [args.scale]  if args.scale  else ["n100", "n200"]

    for family in families:
        for scale in scales:
            instance_dir = Path("data") / family / scale / args.split
            paths = sorted(instance_dir.glob("*.cnf"))
            if not paths:
                log.warning("No instances at %s — skipping", instance_dir)
                continue

            max_flips, max_tries = BUDGETS[scale]
            log.info("==> %s/%s  [%d instances, budget %d flips x %d tries]",
                     family, scale, len(paths), max_flips, max_tries)

            for policy_name, policy in POLICIES:
                results = []
                for i, p in enumerate(paths):
                    formula = parse_dimacs(p)
                    log.debug("[%s] instance %d/%d: %s",
                              policy_name, i + 1, len(paths), p.name)
                    result = solve(formula, policy, max_flips, max_tries)
                    results.append(result)

                    status = "SOLVED " if result.solved else "TIMEOUT"
                    log.info("  %s | %s | %s | %d flips",
                             policy_name, p.name, status, result.n_flips)

                s = summarise(results, max_flips)
                print(
                    f"{family}/{scale} | {policy_name:10s} | "
                    f"solve={s.solve_rate:.0%}  "
                    f"median={s.median_flips:>8.0f}  "
                    f"mean={s.mean_flips:>8.0f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run classical SLS baselines")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Per-instance results (INFO level)")
    parser.add_argument("--debug", action="store_true",
                        help="Per-try detail (DEBUG level)")
    parser.add_argument("--save-logs", action="store_true",
                        help="Mirror output to logs/run_<timestamp>.log")
    parser.add_argument("--family", choices=["kcoloring", "random_3sat"], default=None)
    parser.add_argument("--scale", choices=["n100", "n200"], default=None)
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    setup_logging(
        verbose=args.verbose or args.debug,
        debug=args.debug,
        log_dir=Path("logs") if args.save_logs else None,
    )
    run(args)


if __name__ == "__main__":
    main()
