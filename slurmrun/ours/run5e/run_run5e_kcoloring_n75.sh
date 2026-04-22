#!/bin/bash
#SBATCH --job-name=R5e_KC_75
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=36:00:00
#SBATCH --output=logs/slurm/run5e_kcoloring_n75_%j.out
#SBATCH --error=logs/slurm/run5e_kcoloring_n75_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/models/kcoloring/n75/mlp_base_e
mkdir -p experiments/results/ours/run5e/kcoloring
mkdir -p logs/slurm

echo "===== Run 5e: mlp/base+entropy | kcoloring/n75 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

START=$(date +%s)
python scripts/train.py \
    --family kcoloring \
    --scale n75 \
    --policy mlp \
    --feature-set base \
    --entropy-coef 0.01 \
    --epochs 60 \
    --warmup-epochs 5 \
    --gamma 0.5 \
    --lr 1e-3 \
    --save-dir experiments/models \
    --seed 42 \
    --verbose \
    --wandb
END=$(date +%s)
echo "  Training time: $((END - START))s"

START=$(date +%s)
python scripts/evaluate.py \
    --family kcoloring --scale n75 --split val \
    --policies minbreak noveltyplus mlp \
    --feature-set base \
    --model-path experiments/models/kcoloring/n75/mlp_base_e/best_mlp_base_e.pt \
    --max-tries 10 \
    --save-csv experiments/results/ours/run5e/kcoloring/n75_val.csv
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
