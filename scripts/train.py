"""
Offline training entry point.

Usage:
    python scripts/train.py --config experiments/configs/base.yaml
"""

import argparse
from pathlib import Path
import yaml


def main(config_path: Path) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    main(args.config)
