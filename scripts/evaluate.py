"""
Evaluation entry point. Runs a saved policy (or a classical baseline) against
a test set and reports median flips, solve rate, and CDF curves.

Usage:
    python scripts/evaluate.py --config experiments/configs/base.yaml --policy mlp --checkpoint path/to/weights.pt
    python scripts/evaluate.py --config experiments/configs/base.yaml --policy minbreak
"""

import argparse
from pathlib import Path
import yaml


def main(config_path: Path, policy_name: str, checkpoint: Path | None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", type=str, required=True,
                        choices=["minbreak", "noveltyplus", "linear", "mlp"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    main(args.config, args.policy, args.checkpoint)
