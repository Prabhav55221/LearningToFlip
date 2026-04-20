#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=NP_KCOL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/noveltyplus_kcoloring_%j.out
#SBATCH --error=logs/slurm/noveltyplus_kcoloring_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/results/noveltyplus/kcoloring
mkdir -p logs/slurm

echo "===== NoveltyPlus (p=0.1) | kcoloring ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

# --- n100 val ---
echo ""
echo "-- kcoloring/n100 val --"
START=$(date +%s)
python scripts/evaluate.py \
    --family kcoloring \
    --scale n100 \
    --split val \
    --policies noveltyplus \
    --max-tries 10 \
    --save-csv experiments/results/noveltyplus/kcoloring/n100_val.csv
END=$(date +%s)
echo "  Time: $((END - START))s"

# --- n100 test ---
echo ""
echo "-- kcoloring/n100 test --"
START=$(date +%s)
python scripts/evaluate.py \
    --family kcoloring \
    --scale n100 \
    --split test \
    --policies noveltyplus \
    --max-tries 10 \
    --save-csv experiments/results/noveltyplus/kcoloring/n100_test.csv
END=$(date +%s)
echo "  Time: $((END - START))s"

# --- n200 val ---
echo ""
echo "-- kcoloring/n200 val --"
START=$(date +%s)
python scripts/evaluate.py \
    --family kcoloring \
    --scale n200 \
    --split val \
    --policies noveltyplus \
    --max-tries 10 \
    --save-csv experiments/results/noveltyplus/kcoloring/n200_val.csv
END=$(date +%s)
echo "  Time: $((END - START))s"

# --- n200 test ---
echo ""
echo "-- kcoloring/n200 test --"
START=$(date +%s)
python scripts/evaluate.py \
    --family kcoloring \
    --scale n200 \
    --split test \
    --policies noveltyplus \
    --max-tries 10 \
    --save-csv experiments/results/noveltyplus/kcoloring/n200_test.csv
END=$(date +%s)
echo "  Time: $((END - START))s"

TOTAL_END=$(date +%s)
echo ""
echo "===== Done. Total time: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
