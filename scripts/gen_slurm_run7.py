"""
Generate Slurm scripts for Run 7: MLP / full features / entropy + normalize + noise-walk + reward-normalize.

Writes one script per (family, scale) to slurmrun/ours/run7/.
Usage:
    python scripts/gen_slurm_run7.py
"""

from pathlib import Path

ACCOUNT   = "jeisner1"
CONDA_ENV = "prabhav2"
HOME      = "/home/psingh54"
SCRATCH   = f"{HOME}/scratchjeisner1/psingh54/LearningToFlip"

ENTROPY_COEF = 0.1
EPOCHS       = 60
WARMUP       = 5
LR           = "1e-3"
SEED         = 42

RUN_NAME   = "mlp_full_e_norm_nw_rn"
RUN_LABEL  = "run7"

FAMILIES = {
    "kcoloring":   ["n50", "n75", "n100"],
    "random_3sat": ["n40", "n50", "n60", "n70"],
    "kclique":     ["n5",  "n10", "n15", "n20"],
    "domset":      ["n5",  "n7",  "n9",  "n12"],
}

ABBREVS = {
    "kcoloring":   "KC",
    "random_3sat": "R3",
    "kclique":     "KQ",
    "domset":      "DS",
}

SCALE_NUM = {
    "n5": "5", "n7": "7", "n9": "9",
    "n10": "10", "n12": "12", "n15": "15", "n20": "20",
    "n40": "40", "n50": "50", "n60": "60", "n70": "70",
    "n75": "75", "n100": "100", "n200": "200", "n300": "300",
}


def script(family: str, scale: str) -> str:
    abbrev   = ABBREVS[family]
    num      = SCALE_NUM[scale]
    job_name = f"R7_{abbrev}_{num}"
    log_stem = f"{RUN_LABEL}_{family}_{scale}"
    model_dir   = f"experiments/models/{family}/{scale}/{RUN_NAME}"
    results_dir = f"experiments/results/ours/{RUN_LABEL}/{family}"
    model_path  = f"{model_dir}/best_{RUN_NAME}.pt"

    train_cmd = (
        f"python scripts/train.py \\\n"
        f"    --family {family} \\\n"
        f"    --scale {scale} \\\n"
        f"    --policy mlp \\\n"
        f"    --feature-set full \\\n"
        f"    --entropy-coef {ENTROPY_COEF} \\\n"
        f"    --normalize-features \\\n"
        f"    --noise-walk \\\n"
        f"    --reward-normalize \\\n"
        f"    --epochs {EPOCHS} \\\n"
        f"    --warmup-epochs {WARMUP} \\\n"
        f"    --gamma 0.5 \\\n"
        f"    --lr {LR} \\\n"
        f"    --save-dir experiments/models \\\n"
        f"    --seed {SEED} \\\n"
        f"    --verbose \\\n"
        f"    --wandb"
    )

    eval_cmd = (
        f"python scripts/evaluate.py \\\n"
        f"    --family {family} --scale {scale} --split val \\\n"
        f"    --policies minbreak noveltyplus mlp \\\n"
        f"    --feature-set full \\\n"
        f"    --normalize-features \\\n"
        f"    --noise-walk \\\n"
        f"    --model-path {model_path} \\\n"
        f"    --max-tries 10 \\\n"
        f"    --save-csv {results_dir}/{scale}_val.csv"
    )

    return f"""\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A {ACCOUNT}
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/{log_stem}_%j.out
#SBATCH --error=logs/slurm/{log_stem}_%j.err

source {HOME}/.bashrc
module load anaconda3/2024.02-1
conda activate {CONDA_ENV}
cd {SCRATCH}
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p {model_dir}
mkdir -p {results_dir}
mkdir -p logs/slurm

echo "===== Run 7: mlp/full+entropy+norm+noise_walk+reward_norm | {family}/{scale} ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

START=$(date +%s)
{train_cmd}
END=$(date +%s)
echo "  Training time: $((END - START))s"

START=$(date +%s)
{eval_cmd}
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
"""


def main() -> None:
    out_dir = Path("slurmrun/ours/run7")
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for family, scales in FAMILIES.items():
        for scale in scales:
            content  = script(family, scale)
            fname    = f"run_{RUN_LABEL}_{family}_{scale}.sh"
            fpath    = out_dir / fname
            fpath.write_text(content)
            fpath.chmod(0o755)
            generated.append(str(fpath))
            print(f"  wrote {fpath}")

    print(f"\n{len(generated)} scripts written to {out_dir}/")
    print("\nsbatch all run7 scripts:")
    print(f"  for f in {out_dir}/*.sh; do sbatch $f; done")
    print("\nsbatch run7 (exclude random_3sat):")
    print(
        f"  for f in {out_dir}/*.sh; do "
        "[[ $f != *random_3sat* ]] && sbatch \"$f\"; "
        "done"
    )


if __name__ == "__main__":
    main()
