#!/bin/bash
#SBATCH --job-name=R7_KQ_15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm/run7_kclique_n15_%j.out
#SBATCH --error=logs/slurm/run7_kclique_n15_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/models/kclique/n15/mlp_full_e_kl_sm
mkdir -p experiments/results/ours/run7/kclique
mkdir -p logs/slurm

echo "===== Run 7: mlp/full+entropy+kl+small | kclique/n15 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

START=$(date +%s)
python scripts/train.py \
    --family kclique \
    --scale n15 \
    --policy mlp \
    --feature-set full \
    --hidden-dim 32 \
    --n-layers 1 \
    --entropy-coef 0.01 \
    --kl-anchor-coef 0.01 \
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
    --family kclique --scale n15 --split val \
    --policies minbreak noveltyplus mlp \
    --feature-set full \
    --hidden-dim 32 \
    --n-layers 1 \
    --model-path experiments/models/kclique/n15/mlp_full_e_kl_sm/best_mlp_full_e_kl_sm.pt \
    --max-tries 10 \
    --save-csv experiments/results/ours/run7/kclique/n15_val.csv
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
