#!/bin/bash
#SBATCH --job-name=INT_DS_12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/interian_domset_n12_%j.out
#SBATCH --error=logs/slurm/interian_domset_n12_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/models/domset/n12/interian
mkdir -p experiments/results/interian/domset
mkdir -p logs/slurm

echo "===== Interian | domset/n12 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

START=$(date +%s)
python scripts/train_interian.py \
    --family domset \
    --scale n12 \
    --epochs 60 \
    --warmup-epochs 5 \
    --gamma 0.5 \
    --max-lr 1e-3 \
    --save-dir experiments/models \
    --seed 42 \
    --wandb
END=$(date +%s)
echo "  Training time: $((END - START))s"

START=$(date +%s)
python scripts/evaluate.py \
    --family domset --scale n12 --split val \
    --policies minbreak noveltyplus interian \
    --model-path experiments/models/domset/n12/interian/best_interian.pt \
    --max-tries 10 \
    --save-csv experiments/results/interian/domset/n12_val.csv
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
