#!/bin/bash
#SBATCH --job-name=R4_KC_50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/run4_kcoloring_n50_%j.out
#SBATCH --error=logs/slurm/run4_kcoloring_n50_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/models/kcoloring/n50/linear_base
mkdir -p experiments/results/ours/run4/kcoloring
mkdir -p logs/slurm

echo "===== Run 4: linear/base | kcoloring/n50 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

START=$(date +%s)
python scripts/train.py \
    --family kcoloring \
    --scale n50 \
    --policy linear \
    --feature-set base \
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
    --family kcoloring --scale n50 --split val \
    --policies minbreak noveltyplus linear \
    --feature-set base \
    --model-path experiments/models/kcoloring/n50/linear_base/best_linear_base.pt \
    --max-tries 10 \
    --save-csv experiments/results/ours/run4/kcoloring/n50_val.csv
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
