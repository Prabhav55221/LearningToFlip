#!/bin/bash
#SBATCH --job-name=R7_DS_5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/run7_domset_n5_%j.out
#SBATCH --error=logs/slurm/run7_domset_n5_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/models/domset/n5/mlp_full_e_norm_nw_rn
mkdir -p experiments/results/ours/run7/domset
mkdir -p logs/slurm

echo "===== Run 7: mlp/full+entropy+norm+noise_walk+reward_norm | domset/n5 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

START=$(date +%s)
python scripts/train.py \
    --family domset \
    --scale n5 \
    --policy mlp \
    --feature-set full \
    --entropy-coef 0.1 \
    --normalize-features \
    --noise-walk \
    --reward-normalize \
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
    --family domset --scale n5 --split val \
    --policies minbreak noveltyplus mlp \
    --feature-set full \
    --normalize-features \
    --noise-walk \
    --model-path experiments/models/domset/n5/mlp_full_e_norm_nw_rn/best_mlp_full_e_norm_nw_rn.pt \
    --max-tries 10 \
    --save-csv experiments/results/ours/run7/domset/n5_val.csv
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
