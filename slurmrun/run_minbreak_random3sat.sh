#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=MB_3SAT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/minbreak_random3sat_%j.out
#SBATCH --error=logs/slurm/minbreak_random3sat_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/results/minbreak/random_3sat
mkdir -p logs/slurm

echo "===== MinBreak | random_3sat ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

# --- n100 val ---
echo ""
echo "-- random_3sat/n100 val --"
START=$(date +%s)
python scripts/evaluate.py \
    --family random_3sat \
    --scale n100 \
    --split val \
    --policies minbreak \
    --max-tries 10 \
    --save-csv experiments/results/minbreak/random_3sat/n100_val.csv
END=$(date +%s)
echo "  Time: $((END - START))s"

# --- n100 test ---
echo ""
echo "-- random_3sat/n100 test --"
START=$(date +%s)
python scripts/evaluate.py \
    --family random_3sat \
    --scale n100 \
    --split test \
    --policies minbreak \
    --max-tries 10 \
    --save-csv experiments/results/minbreak/random_3sat/n100_test.csv
END=$(date +%s)
echo "  Time: $((END - START))s"

# --- n200 val ---
echo ""
echo "-- random_3sat/n200 val --"
START=$(date +%s)
python scripts/evaluate.py \
    --family random_3sat \
    --scale n200 \
    --split val \
    --policies minbreak \
    --max-tries 10 \
    --save-csv experiments/results/minbreak/random_3sat/n200_val.csv
END=$(date +%s)
echo "  Time: $((END - START))s"

# --- n200 test ---
echo ""
echo "-- random_3sat/n200 test --"
START=$(date +%s)
python scripts/evaluate.py \
    --family random_3sat \
    --scale n200 \
    --split test \
    --policies minbreak \
    --max-tries 10 \
    --save-csv experiments/results/minbreak/random_3sat/n200_test.csv
END=$(date +%s)
echo "  Time: $((END - START))s"

TOTAL_END=$(date +%s)
echo ""
echo "===== Done. Total time: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
