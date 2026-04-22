#!/bin/bash
#SBATCH --job-name=BL_DS_12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/baselines_domset_n12_%j.out
#SBATCH --error=logs/slurm/baselines_domset_n12_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/results/baselines/domset
mkdir -p logs/slurm

echo "===== Baselines | domset/n12 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

START=$(date +%s)
python scripts/evaluate.py \
    --family domset --scale n12 --split val \
    --policies minbreak noveltyplus \
    --max-tries 10 \
    --save-csv experiments/results/baselines/domset/n12_val.csv
END=$(date +%s)
echo "  val time: $((END - START))s"

START=$(date +%s)
python scripts/evaluate.py \
    --family domset --scale n12 --split test \
    --policies minbreak noveltyplus \
    --max-tries 10 \
    --save-csv experiments/results/baselines/domset/n12_test.csv
END=$(date +%s)
echo "  test time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
