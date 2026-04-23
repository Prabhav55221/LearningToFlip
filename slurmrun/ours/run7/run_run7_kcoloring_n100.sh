#!/bin/bash
#SBATCH --job-name=R7_KC_100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=60:00:00
#SBATCH --output=logs/slurm/run7_kcoloring_n100_%j.out
#SBATCH --error=logs/slurm/run7_kcoloring_n100_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p experiments/models/kcoloring/n100/mlp_full_e_norm_fw_sm
mkdir -p experiments/results/ours/run7/kcoloring
mkdir -p logs/slurm

echo "===== Run 7: mlp/full+entropy+norm+fixed_walk(p=0.0895)+small | kcoloring/n100 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

START=$(date +%s)
python scripts/train.py \
    --family kcoloring \
    --scale n100 \
    --policy mlp \
    --feature-set full \
    --hidden-dim 16 \
    --n-layers 1 \
    --entropy-coef 0.1 \
    --normalize-features \
    --noise-prob 0.0895 \
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
    --family kcoloring --scale n100 --split val \
    --policies minbreak noveltyplus mlp \
    --feature-set full \
    --normalize-features \
    --hidden-dim 16 \
    --n-layers 1 \
    --noise-prob 0.0895 \
    --model-path experiments/models/kcoloring/n100/mlp_full_e_norm_fw_sm/best_mlp_full_e_norm_fw_sm.pt \
    --max-tries 10 \
    --save-csv experiments/results/ours/run7/kcoloring/n100_val.csv
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
