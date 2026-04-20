#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=NP_KCOL_50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/noveltyplus_kcoloring_n50_%j.out
#SBATCH --error=logs/slurm/noveltyplus_kcoloring_n50_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/results/noveltyplus/kcoloring
mkdir -p logs/slurm

echo "===== NoveltyPlus (p=0.1) | kcoloring/n50 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

echo ""
echo "-- kcoloring/n50 val --"
START=$(date +%s)
python scripts/evaluate.py \
    --family kcoloring --scale n50 --split val \
    --policies noveltyplus --max-tries 10 \
    --save-csv experiments/results/noveltyplus/kcoloring/n50_val.csv
END=$(date +%s)
echo "  Time: $((END - START))s"

echo ""
echo "-- kcoloring/n50 test --"
START=$(date +%s)
python scripts/evaluate.py \
    --family kcoloring --scale n50 --split test \
    --policies noveltyplus --max-tries 10 \
    --save-csv experiments/results/noveltyplus/kcoloring/n50_test.csv
END=$(date +%s)
echo "  Time: $((END - START))s"

TOTAL_END=$(date +%s)
echo ""
echo "===== Done. Total time: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
