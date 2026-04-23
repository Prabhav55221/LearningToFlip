#!/bin/bash
#SBATCH --job-name=CROSS_KC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/cross_kcoloring_to_others_%j.out
#SBATCH --error=logs/slurm/cross_kcoloring_to_others_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p logs/slurm
mkdir -p experiments/generalization/cross/kcoloring_to_domset
mkdir -p experiments/generalization/cross/kcoloring_to_kclique

echo "===== Cross-domain H3: kcoloring/n50 model → other families ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

echo "--- Generating missing kclique n12 test data ---"
bash scripts/generate_gen_data.sh kclique

# --- kcoloring/n50 → domset/n7 ---
START=$(date +%s)
echo "--- kcoloring model on domset/n7 instances ---"
python scripts/eval_generalization.py \
    --family domset \
    --train-scale n50 \
    --model-path experiments/models/kcoloring/n50/mlp_full_e_kl_sm/best_mlp_full_e_kl_sm.pt \
    --interian-model experiments/models/domset/n7/interian/best_interian.pt \
    --model-label cross_kcoloring_n50 \
    --split test \
    --test-scales n7 \
    --max-tries 10 \
    --max-instances 100 \
    --save-dir experiments/generalization/cross/kcoloring_to_domset \
    --methods minbreak noveltyplus interian ours_frozen ours_online_kl ours_online_success_kl \
    --verbose
END=$(date +%s)
echo "  kcoloring→domset time: $((END - START))s"

# --- kcoloring/n50 → kclique/n12 ---
START=$(date +%s)
echo "--- kcoloring model on kclique/n12 instances ---"
python scripts/eval_generalization.py \
    --family kclique \
    --train-scale n50 \
    --model-path experiments/models/kcoloring/n50/mlp_full_e_kl_sm/best_mlp_full_e_kl_sm.pt \
    --interian-model experiments/models/kclique/n5/interian/best_interian.pt \
    --model-label cross_kcoloring_n50 \
    --split test \
    --test-scales n12 \
    --max-tries 10 \
    --max-instances 100 \
    --save-dir experiments/generalization/cross/kcoloring_to_kclique \
    --methods minbreak noveltyplus interian ours_frozen ours_online_kl ours_online_success_kl \
    --verbose
END=$(date +%s)
echo "  kcoloring→kclique time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
