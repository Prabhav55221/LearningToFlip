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
    parser.add_argument("--family",             choices=["kcoloring", "random_3sat", "kclique", "domset"], required=True)
    parser.add_argument("--scale",              choices=list(BUDGETS.keys()), required=True,
                        help="Primary scale (used for save path; also the only training scale unless --scales is given).")
    parser.add_argument("--scales",             nargs="+", choices=list(BUDGETS.keys()), default=None,
                        help="Multi-scale training: load data from all listed scales. Overrides --scale for data loading.")
    parser.add_argument("--policy",             choices=["linear", "mlp"],          default="mlp")
    parser.add_argument("--feature-set",        choices=list(FEATURE_SETS.keys()),  default="full")
    parser.add_argument("--hidden-dim",         type=int, default=64,
                        help="MLP hidden layer width.")
    parser.add_argument("--n-layers",           type=int, default=2,
                        help="Number of MLP hidden layers.")
    parser.add_argument("--noise-prob",         type=float, default=0.0,
                        help="Fixed random-walk probability per step (Interian escape mechanism).")
    parser.add_argument("--normalize-features", action="store_true",
                        help="Normalize count features by avg_deg and time features by step.")
    parser.add_argument("--k",             type=int,   default=10)
    parser.add_argument("--gamma",         type=float, default=0.5)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--epochs",        type=int,   default=60)
    parser.add_argument("--warmup-epochs", type=int,   default=5)
    parser.add_argument("--save-dir",      type=Path,  default=Path("experiments/models"))
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--entropy-coef",  type=float, default=0.0, help="Entropy regularization coefficient (0=disabled).")
    parser.add_argument("--verbose",       action="store_true")
    parser.add_argument("--wandb",         action="store_true", help="Enable Weights & Biases logging.")
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Resolve scales: multi-scale overrides single scale for data loading
    scales = args.scales if args.scales else [args.scale]
    scale_str = "-".join(scales) if len(scales) > 1 else args.scale
    max_flips = max(BUDGETS[s] for s in scales)

    run_name = f"{args.policy}_{args.feature_set}"
    if args.entropy_coef > 0:
        run_name += "_e"
    if len(scales) > 1:
        run_name += "_ms"
    if args.normalize_features:
        run_name += "_norm"
    if args.noise_prob > 0.0:
        run_name += "_fw"
    if args.hidden_dim < 64 or args.n_layers < 2:
        run_name += "_sm"

    log_fn = None
    if args.wandb:
        import wandb
        wandb.init(
            project="LearningToFlip",
            name=f"{run_name}_{args.family}_{scale_str}_s{args.seed}",
            config={
                "method": "ours",
                "policy": args.policy,
                "feature_set": args.feature_set,
                "family": args.family,
                "scale": args.scale,
                "scales": scales,
                "normalize_features": args.normalize_features,
                "k": args.k,
                "gamma": args.gamma,
                "lr": args.lr,
                "entropy_coef": args.entropy_coef,
                "epochs": args.epochs,
                "warmup_epochs": args.warmup_epochs,
                "seed": args.seed,
            },
            tags=["ours", args.policy, args.feature_set, args.family, args.scale,
                  *( ["entropy"]    if args.entropy_coef > 0   else []),
                  *( ["multiscale"] if len(scales) > 1         else []),
                  *( ["normalize"]  if args.normalize_features else []),
                  *( ["fixed_walk"] if args.noise_prob > 0.0   else []),
                  *( ["small_mlp"]  if args.hidden_dim < 64    else [])],
        )
        log_fn = wandb.log

    train_paths: list[Path] = []
    val_paths:   list[Path] = []
    for sc in scales:
        data_root = Path("data") / args.family / sc
        train_paths += sorted((data_root / "train").glob("*.cnf"))
        val_paths   += sorted((data_root / "val").glob("*.cnf"))
    if not train_paths:
        raise FileNotFoundError(f"No .cnf train files found for scales {scales} in {args.family}")
    if not val_paths:
        raise FileNotFoundError(f"No .cnf val files found for scales {scales} in {args.family}")

    log.info(
        "=== %s/%s | policy=%s | feature_set=%s | k=%d gamma=%.2f lr=%.4f normalize=%s ===",
        args.family, scale_str, args.policy, args.feature_set,
        args.k, args.gamma, args.lr, args.normalize_features,
    )

    policy = (
        LinearPolicy(feature_set=args.feature_set, normalize=args.normalize_features)
        if args.policy == "linear"
        else MLPPolicy(
            feature_set=args.feature_set,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            normalize=args.normalize_features,
            noise_prob=args.noise_prob,
        )
    )

    config = REINFORCEConfig(
        k=args.k,
        gamma=args.gamma,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        max_flips=max_flips,
    )

    save_dir = args.save_dir / args.family / scale_str / run_name

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
