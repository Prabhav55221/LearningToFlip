#!/bin/bash
#SBATCH --job-name=BL_KCL_20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=8GB
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/baselines_kclique_n20_%j.out
#SBATCH --error=logs/slurm/baselines_kclique_n20_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/results/baselines/kclique
mkdir -p logs/slurm

echo "===== Baselines | kclique/n20 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

START=$(date +%s)
python scripts/evaluate.py \
    --family kclique --scale n20 --split val \
    --policies minbreak noveltyplus \
    --max-tries 10 \
    --save-csv experiments/results/baselines/kclique/n20_val.csv
END=$(date +%s)
echo "  val time: $((END - START))s"

START=$(date +%s)
python scripts/evaluate.py \
    --family kclique --scale n20 --split test \
    --policies minbreak noveltyplus \
    --max-tries 10 \
    --save-csv experiments/results/baselines/kclique/n20_test.csv
END=$(date +%s)
echo "  test time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
