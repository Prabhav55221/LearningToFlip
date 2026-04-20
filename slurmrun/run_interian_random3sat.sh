#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=INT_3SAT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=18GB
#SBATCH --gpus=1
#SBATCH --partition=a100
#SBATCH --exclude=c001
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/interian_random3sat_%j.out
#SBATCH --error=logs/slurm/interian_random3sat_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/models/random_3sat/n100/interian
mkdir -p experiments/models/random_3sat/n200/interian
mkdir -p experiments/results/interian/random_3sat
mkdir -p logs/slurm

echo "===== Interian REINFORCE | random_3sat ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

# ------------------------------------------------------------------ #
# Train — n100                                                         #
# ------------------------------------------------------------------ #
echo ""
echo "== Training: random_3sat/n100 =="
START=$(date +%s)
python scripts/train_interian.py \
    --family random_3sat \
    --scale n100 \
    --epochs 60 \
    --warmup-epochs 5 \
    --gamma 0.5 \
    --max-lr 1e-3 \
    --save-dir experiments/models \
    --seed 42
END=$(date +%s)
echo "  Training time: $((END - START))s"

# Evaluate n100 val (includes minbreak + noveltyplus for context)
echo ""
echo "-- Eval: random_3sat/n100 val --"
START=$(date +%s)
python scripts/evaluate.py \
    --family random_3sat \
    --scale n100 \
    --split val \
    --policies minbreak noveltyplus interian \
    --model-path experiments/models/random_3sat/n100/interian/best_interian.pt \
    --max-tries 10 \
    --save-csv experiments/results/interian/random_3sat/n100_val.csv
END=$(date +%s)
echo "  Eval time: $((END - START))s"

# ------------------------------------------------------------------ #
# Train — n200                                                         #
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

# Evaluate n200 val
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
    --save-csv experiments/results/interian/random_3sat/n200_val.csv
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo ""
echo "===== Done. Total time: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
