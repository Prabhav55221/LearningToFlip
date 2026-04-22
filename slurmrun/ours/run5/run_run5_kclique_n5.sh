#!/bin/bash
#SBATCH --job-name=R5_KCL_5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm/run5_kclique_n5_%j.out
#SBATCH --error=logs/slurm/run5_kclique_n5_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/models/kclique/n5/mlp_base
mkdir -p experiments/results/ours/run5/kclique
mkdir -p logs/slurm

echo "===== Run 5: mlp/base | kclique/n5 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

START=$(date +%s)
python scripts/train.py \
    --family kclique \
    --scale n5 \
    --policy mlp \
    --feature-set base \
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
    --family kclique --scale n5 --split val \
    --policies minbreak noveltyplus mlp \
    --feature-set base \
    --model-path experiments/models/kclique/n5/mlp_base/best_mlp_base.pt \
    --max-tries 10 \
    --save-csv experiments/results/ours/run5/kclique/n5_val.csv
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
