#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=INT_3SAT_200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/interian_random3sat_n200_%j.out
#SBATCH --error=logs/slurm/interian_random3sat_n200_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/models/random_3sat/n200/interian
mkdir -p experiments/results/interian/random_3sat
mkdir -p logs/slurm

echo "===== Interian REINFORCE | random_3sat/n200 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

# ------------------------------------------------------------------ #
# Train                                                                #
# ------------------------------------------------------------------ #
echo ""
echo "== Training: random_3sat/n200 =="
START=$(date +%s)
python scripts/train_interian.py \
    --family random_3sat \
    --scale n200 \
    --epochs 60 \
    --warmup-epochs 5 \
    --gamma 0.5 \
    --max-lr 1e-3 \
    --save-dir experiments/models \
    --seed 42
END=$(date +%s)
echo "  Training time: $((END - START))s"

# ------------------------------------------------------------------ #
# Evaluate (val split — test stays locked)                            #
# ------------------------------------------------------------------ #
echo ""
echo "-- Eval: random_3sat/n200 val --"
START=$(date +%s)
python scripts/evaluate.py \
    --family random_3sat \
    --scale n200 \
    --split val \
    --policies minbreak noveltyplus interian \
    --model-path experiments/models/random_3sat/n200/interian/best_interian.pt \
    --max-tries 10 \
    --verbose \
    --save-csv experiments/results/interian/random_3sat/n200_val.csv
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo ""
echo "===== Done. Total time: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
