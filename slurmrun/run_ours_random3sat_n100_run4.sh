#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=R4_3SAT_100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/ours_random3sat_n100_run4_%j.out
#SBATCH --error=logs/slurm/ours_random3sat_n100_run4_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/models/random_3sat/n100/linear_base
mkdir -p experiments/results/ours/random_3sat
mkdir -p logs/slurm

echo "===== Run 4: linear/base | random_3sat/n100 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

echo ""
echo "== Training: linear/base random_3sat/n100 =="
START=$(date +%s)
python scripts/train.py \
    --family random_3sat \
    --scale n100 \
    --policy linear \
    --feature-set base \
    --epochs 60 \
    --warmup-epochs 5 \
    --gamma 0.5 \
    --lr 1e-3 \
    --save-dir experiments/models \
    --seed 42 \
    --verbose
END=$(date +%s)
echo "  Training time: $((END - START))s"

echo ""
echo "-- Eval: linear/base random_3sat/n100 val --"
START=$(date +%s)
python scripts/evaluate.py \
    --family random_3sat \
    --scale n100 \
    --split val \
    --policies minbreak noveltyplus interian linear \
    --feature-set base \
    --model-path experiments/models/random_3sat/n100/linear_base/best_linear_base.pt \
    --max-tries 10 \
    --verbose \
    --save-csv experiments/results/ours/random_3sat/n100_val_run4.csv
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo ""
echo "===== Done. Total time: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
