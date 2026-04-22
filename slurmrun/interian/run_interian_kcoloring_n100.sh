#!/bin/bash
#SBATCH --job-name=INT_KC_100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/interian_kcoloring_n100_%j.out
#SBATCH --error=logs/slurm/interian_kcoloring_n100_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/models/kcoloring/n100/interian
mkdir -p experiments/results/interian/kcoloring
mkdir -p logs/slurm

echo "===== Interian | kcoloring/n100 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

START=$(date +%s)
python scripts/train_interian.py \
    --family kcoloring \
    --scale n100 \
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
    --family kcoloring --scale n100 --split val \
    --policies minbreak noveltyplus interian \
    --model-path experiments/models/kcoloring/n100/interian/best_interian.pt \
    --max-tries 10 \
    --save-csv experiments/results/interian/kcoloring/n100_val.csv
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
