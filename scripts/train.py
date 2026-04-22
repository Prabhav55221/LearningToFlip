"""
Offline training entry point for our REINFORCE method (runs 4, 5, 6).

Usage:
    python scripts/train.py --family kcoloring --scale n50 \
        --policy mlp --feature-set full --wandb

Checkpoint: experiments/models/{family}/{scale}/{policy}_{feature_set}/best_{run_name}.pt
W&B run name: {policy}_{feature_set}_{family}_{scale}_s{seed}
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

BUDGETS = {
    "n5":   500,   "n7":   700,   "n9":   900,
    "n10": 1_000,  "n12": 1_200,  "n15": 1_500,  "n20": 2_000,
    "n40": 4_000,  "n50": 5_000,  "n60": 6_000,  "n70": 7_000,
    "n75": 7_500,  "n100": 10_000, "n200": 20_000, "n300": 30_000,
}

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train our REINFORCE policy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--family",        choices=["kcoloring", "random_3sat", "kclique", "domset"], required=True)
    parser.add_argument("--scale",         choices=list(BUDGETS.keys()),    required=True)
    parser.add_argument("--policy",        choices=["linear", "mlp"],       default="mlp")
    parser.add_argument("--feature-set",   choices=list(FEATURE_SETS.keys()), default="full")
    parser.add_argument("--k",             type=int,   default=10)
    parser.add_argument("--gamma",         type=float, default=0.5)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--epochs",        type=int,   default=60)
    parser.add_argument("--warmup-epochs", type=int,   default=5)
    parser.add_argument("--save-dir",      type=Path,  default=Path("experiments/models"))
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--verbose",       action="store_true")
    parser.add_argument("--wandb",         action="store_true", help="Enable Weights & Biases logging.")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = f"{args.policy}_{args.feature_set}"

    log_fn = None
    if args.wandb:
        import wandb
        wandb.init(
            project="LearningToFlip",
            name=f"{run_name}_{args.family}_{args.scale}_s{args.seed}",
            config={
                "method": "ours",
                "policy": args.policy,
                "feature_set": args.feature_set,
                "family": args.family,
                "scale": args.scale,
                "k": args.k,
                "gamma": args.gamma,
                "lr": args.lr,
                "epochs": args.epochs,
                "warmup_epochs": args.warmup_epochs,
                "seed": args.seed,
            },
            tags=["ours", args.policy, args.feature_set, args.family, args.scale],
        )
        log_fn = wandb.log

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

    policy = LinearPolicy(feature_set=args.feature_set) if args.policy == "linear" \
        else MLPPolicy(feature_set=args.feature_set)

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
        log_fn=log_fn,
    )

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
