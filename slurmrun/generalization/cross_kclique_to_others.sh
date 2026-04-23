#!/bin/bash
#SBATCH --job-name=CROSS_KCL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/cross_kclique_to_others_%j.out
#SBATCH --error=logs/slurm/cross_kclique_to_others_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p logs/slurm
mkdir -p experiments/generalization/cross/kclique_to_kcoloring
mkdir -p experiments/generalization/cross/kclique_to_domset

echo "===== Cross-domain H3: kclique/n5 model → other families ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

# All target scales (kcoloring/n50, domset/n7) have full test splits — no data generation needed.

# --- kclique/n5 → kcoloring/n50 ---
START=$(date +%s)
echo "--- kclique model on kcoloring/n50 instances ---"
python scripts/eval_generalization.py \
    --family kcoloring \
    --train-scale n5 \
    --model-path experiments/models/kclique/n5/mlp_full_e_kl_sm/best_mlp_full_e_kl_sm.pt \
    --interian-model experiments/models/kcoloring/n50/interian/best_interian.pt \
    --model-label cross_kclique_n5 \
    --split test \
    --test-scales n50 \
    --max-tries 10 \
    --max-instances 100 \
    --save-dir experiments/generalization/cross/kclique_to_kcoloring \
    --methods minbreak noveltyplus interian ours_frozen ours_online_kl ours_online_success_kl \
    --verbose
END=$(date +%s)
echo "  kclique→kcoloring time: $((END - START))s"

# --- kclique/n5 → domset/n7 ---
START=$(date +%s)
echo "--- kclique model on domset/n7 instances ---"
python scripts/eval_generalization.py \
    --family domset \
    --train-scale n5 \
    --model-path experiments/models/kclique/n5/mlp_full_e_kl_sm/best_mlp_full_e_kl_sm.pt \
    --interian-model experiments/models/domset/n7/interian/best_interian.pt \
    --model-label cross_kclique_n5 \
    --split test \
    --test-scales n7 \
    --max-tries 10 \
    --max-instances 100 \
    --save-dir experiments/generalization/cross/kclique_to_domset \
    --methods minbreak noveltyplus interian ours_frozen ours_online_kl ours_online_success_kl \
    --verbose
END=$(date +%s)
echo "  kclique→domset time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
