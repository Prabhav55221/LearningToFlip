"""
Training entry point for Interian & Bernardini (KR 2023) replication.

Usage:
    python scripts/train_interian.py --family kcoloring --scale n100 --wandb
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from src.utils.logging import setup as setup_logging
from src.train.interian_reinforce import InterianConfig, train

BUDGETS = {
    "n5":   500,   "n7":   700,   "n9":   900,
    "n10": 1_000,  "n12": 1_200,  "n15": 1_500,  "n20": 2_000,
    "n40": 4_000,  "n50": 5_000,  "n60": 6_000,  "n70": 7_000,
    "n75": 7_500,  "n100": 10_000, "n200": 20_000, "n300": 30_000,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Interian linear policy via REINFORCE")
    parser.add_argument("--family",        choices=["kcoloring", "random_3sat", "kclique", "domset"], required=True)
    parser.add_argument("--scale",         choices=list(BUDGETS.keys()), required=True)
    parser.add_argument("--epochs",        type=int,   default=60)
    parser.add_argument("--warmup-epochs", type=int,   default=5)
    parser.add_argument("--gamma",         type=float, default=0.5)
    parser.add_argument("--max-lr",        type=float, default=1e-3)
    parser.add_argument("--save-dir",      type=Path,  default=Path("experiments/models"))
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--wandb",         action="store_true", help="Enable Weights & Biases logging.")
    args = parser.parse_args()

    setup_logging(verbose=True)
    log = logging.getLogger(__name__)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_fn = None
    if args.wandb:
        import wandb
        wandb.init(
            project="LearningToFlip",
            name=f"interian_{args.family}_{args.scale}_s{args.seed}",
            config={
                "method": "interian",
                "family": args.family,
                "scale": args.scale,
                "epochs": args.epochs,
                "warmup_epochs": args.warmup_epochs,
                "gamma": args.gamma,
                "max_lr": args.max_lr,
                "seed": args.seed,
            },
            tags=["interian", args.family, args.scale],
        )
        log_fn = wandb.log

    data_dir = Path("data") / args.family / args.scale
    train_paths = sorted((data_dir / "train").glob("*.cnf"))
    val_paths   = sorted((data_dir / "val").glob("*.cnf"))

    if not train_paths:
        raise FileNotFoundError(f"No training instances found at {data_dir / 'train'}")
    log.info("Family: %s/%s  |  train=%d  val=%d",
             args.family, args.scale, len(train_paths), len(val_paths))

    config = InterianConfig(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        gamma=args.gamma,
        max_lr=args.max_lr,
        max_flips=BUDGETS[args.scale],
    )

    save_dir = args.save_dir / args.family / args.scale / "interian"
    policy = train(train_paths, val_paths, config, save_dir=save_dir, log_fn=log_fn)

    log.info("Training complete.")
    log.info("  Learned noise prob p_w = %.4f", policy.noise_prob)
    log.info("  Scoring weights:  %s", policy.linear.weight.data.tolist())
    log.info("  Bias:             %.4f", policy.linear.bias.item())

    if args.wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
