#!/bin/bash
#SBATCH --job-name=GEN_DS_7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/gen_domset_from_n7_%j.out
#SBATCH --error=logs/slurm/gen_domset_from_n7_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p logs/slurm
mkdir -p experiments/generalization/domset_from_n7

echo "===== Generalization: domset model trained on n7 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

# All domset scales (n5, n7, n9, n12) have full train/val/test splits — no data generation needed.

START=$(date +%s)
echo "--- Running generalization eval (n5 → n12) ---"
python scripts/eval_generalization.py \
    --family domset \
    --train-scale n7 \
    --model-path experiments/models/domset/n7/mlp_full_e_kl_sm/best_mlp_full_e_kl_sm.pt \
    --interian-model experiments/models/domset/n7/interian/best_interian.pt \
    --model-label run7 \
    --split test \
    --test-scales n5 n7 n9 n12 \
    --max-tries 10 \
    --max-instances 100 \
    --save-dir experiments/generalization/domset_from_n7 \
    --methods minbreak noveltyplus interian ours_frozen ours_online_kl ours_online_success_kl \
    --verbose
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
