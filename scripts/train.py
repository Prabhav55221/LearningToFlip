"""
Offline training entry point for our REINFORCE method (runs 4, 5, 6).

Usage:
    # Run 4: linear policy, base features
    python scripts/train.py --family kcoloring --scale n50 \
        --policy linear --feature-set base --save-dir experiments/models

    # Run 5: MLP, base features
    python scripts/train.py --family random_3sat --scale n100 \
        --policy mlp --feature-set base --save-dir experiments/models

    # Run 6: MLP, full features
    python scripts/train.py --family kcoloring --scale n100 \
        --policy mlp --feature-set full --save-dir experiments/models

Checkpoint saved to:
    {save-dir}/{family}/{scale}/{policy}_{feature-set}/best_{policy}_{feature-set}.pt
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from src.utils.logging import setup as setup_logging
from src.policy.linear import LinearPolicy
from src.policy.mlp import MLPPolicy
from src.policy.features import FEATURE_SETS
from src.train.reinforce import REINFORCEConfig, train

BUDGETS = {"n50": 5_000, "n100": 10_000, "n200": 50_000}

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train our REINFORCE policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--family",        choices=["kcoloring", "random_3sat"], required=True)
    parser.add_argument("--scale",         choices=["n50", "n100", "n200"],      required=True)
    parser.add_argument("--policy",        choices=["linear", "mlp"],            default="mlp")
    parser.add_argument("--feature-set",   choices=list(FEATURE_SETS.keys()),    default="full")
    parser.add_argument("--k",             type=int,   default=10)
    parser.add_argument("--gamma",         type=float, default=0.5)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--epochs",        type=int,   default=60)
    parser.add_argument("--warmup-epochs", type=int,   default=5)
    parser.add_argument("--save-dir",      type=Path,  default=Path("experiments/models"))
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = f"{args.policy}_{args.feature_set}"

    data_root = Path("data") / args.family / args.scale
    train_paths = sorted((data_root / "train").glob("*.cnf"))
    val_paths   = sorted((data_root / "val").glob("*.cnf"))
    if not train_paths:
        raise FileNotFoundError(f"No .cnf files in {data_root / 'train'}")
    if not val_paths:
        raise FileNotFoundError(f"No .cnf files in {data_root / 'val'}")

    log.info(
        "=== %s/%s | policy=%s | feature_set=%s | k=%d gamma=%.2f lr=%.4f ===",
        args.family, args.scale, args.policy, args.feature_set,
        args.k, args.gamma, args.lr,
    )

    if args.policy == "linear":
        policy = LinearPolicy(feature_set=args.feature_set)
    else:
        policy = MLPPolicy(feature_set=args.feature_set)

    config = REINFORCEConfig(
        k=args.k,
        gamma=args.gamma,
        lr=args.lr,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        max_flips=BUDGETS[args.scale],
    )

    save_dir = args.save_dir / args.family / args.scale / run_name

    train(
        train_paths=train_paths,
        val_paths=val_paths,
        policy=policy,
        config=config,
        save_dir=save_dir,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
