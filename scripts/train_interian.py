"""
Training entry point for Interian & Bernardini (KR 2023) replication.

Usage:
    python scripts/train_interian.py --family kcoloring --scale n100
    python scripts/train_interian.py --family random_3sat --scale n100 --epochs 60
    python scripts/train_interian.py --family kcoloring --scale n100 --split train
"""

import argparse
import logging
from pathlib import Path

import torch

from src.utils.logging import setup as setup_logging
from src.train.interian_reinforce import InterianConfig, train


BUDGETS = {"n50": 5_000, "n100": 10_000, "n200": 50_000}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Interian linear policy via REINFORCE")
    parser.add_argument("--family", choices=["kcoloring", "random_3sat"], required=True)
    parser.add_argument("--scale",  choices=["n50", "n100", "n200"], required=True)
    parser.add_argument("--epochs",        type=int,   default=60)
    parser.add_argument("--warmup-epochs", type=int,   default=5)
    parser.add_argument("--gamma",         type=float, default=0.5)
    parser.add_argument("--max-lr",        type=float, default=1e-3)
    parser.add_argument("--save-dir",      type=Path,
                        default=Path("experiments/models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(verbose=True)
    log = logging.getLogger(__name__)

    torch.manual_seed(args.seed)

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
    policy = train(train_paths, val_paths, config, save_dir=save_dir)

    log.info("Training complete.")
    log.info("  Learned noise prob p_w = %.4f", policy.noise_prob)
    log.info("  Scoring weights:  %s", policy.linear.weight.data.tolist())
    log.info("  Bias:             %.4f", policy.linear.bias.item())


if __name__ == "__main__":
    main()
