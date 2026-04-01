"""
Instance generation — wrapper around CNFgen.

Generates satisfiable SAT instances for all configured families and scales,
writing them to data/{family}/{scale}/instance_{i:04d}.cnf.

Usage:
    python scripts/generate.py --config experiments/configs/base.yaml
"""

import argparse
import subprocess
from pathlib import Path


FAMILIES = {
    "random_3sat": {
        "n100": "cnfgen randkcnf 3 100 426",
        "n200": "cnfgen randkcnf 3 200 852",
    },
    "kcoloring": {
        "n100": "cnfgen kcolor 5 gnp 20 0.5",
        "n200": "cnfgen kcolor 5 gnp 40 0.5",
    },
}

SPLITS = {"train": 1900, "val": 100, "test": 500}


def generate_instance(cnfgen_cmd: str, out_path: Path) -> bool:
    """Run a cnfgen command and write the output .cnf to out_path."""
    raise NotImplementedError


def main(data_dir: Path = Path("data")) -> None:
    for family, scales in FAMILIES.items():
        for scale, cmd in scales.items():
            for split, n in SPLITS.items():
                out_dir = data_dir / family / scale / split
                out_dir.mkdir(parents=True, exist_ok=True)
                for i in range(n):
                    out_path = out_dir / f"instance_{i:04d}.cnf"
                    if not out_path.exists():
                        generate_instance(cmd, out_path)
                print(f"  {family}/{scale}/{split}: {n} instances")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    main(args.data_dir)
