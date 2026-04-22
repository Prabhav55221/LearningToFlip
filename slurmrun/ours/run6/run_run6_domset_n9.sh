#!/bin/bash
#SBATCH --job-name=R6_DS_9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm/run6_domset_n9_%j.out
#SBATCH --error=logs/slurm/run6_domset_n9_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/models/domset/n9/mlp_full
mkdir -p experiments/results/ours/run6/domset
mkdir -p logs/slurm

echo "===== Run 6: mlp/full | domset/n9 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

START=$(date +%s)
python scripts/train.py \
    --family domset \
    --scale n9 \
    --policy mlp \
    --feature-set full \
    --epochs 60 \
    --warmup-epochs 5 \
    --gamma 0.5 \
    --lr 1e-3 \
    --save-dir experiments/models \
    --seed 42 \
    --verbose \
    --wandb
END=$(date +%s)
echo "  Training time: $((END - START))s"

START=$(date +%s)
python scripts/evaluate.py \
    --family domset --scale n9 --split val \
    --policies minbreak noveltyplus mlp \
    --feature-set full \
    --model-path experiments/models/domset/n9/mlp_full/best_mlp_full.pt \
    --max-tries 10 \
    --save-csv experiments/results/ours/run6/domset/n9_val.csv
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
