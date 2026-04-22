#!/bin/bash
#SBATCH --job-name=BL_R3_60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/baselines_random_3sat_n60_%j.out
#SBATCH --error=logs/slurm/baselines_random_3sat_n60_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/results/baselines/random_3sat
mkdir -p logs/slurm

echo "===== Baselines | random_3sat/n60 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

START=$(date +%s)
python scripts/evaluate.py \
    --family random_3sat --scale n60 --split val \
    --policies minbreak noveltyplus \
    --max-tries 10 \
    --save-csv experiments/results/baselines/random_3sat/n60_val.csv
END=$(date +%s)
echo "  val time: $((END - START))s"

START=$(date +%s)
python scripts/evaluate.py \
    --family random_3sat --scale n60 --split test \
    --policies minbreak noveltyplus \
    --max-tries 10 \
    --save-csv experiments/results/baselines/random_3sat/n60_test.csv
END=$(date +%s)
echo "  test time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
