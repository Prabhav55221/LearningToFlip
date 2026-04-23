#!/bin/bash
#SBATCH --job-name=GEN_KCL_5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=10:00:00
#SBATCH --output=logs/slurm/gen_kclique_from_n5_%j.out
#SBATCH --error=logs/slurm/gen_kclique_from_n5_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p logs/slurm
mkdir -p experiments/generalization/kclique_from_n5

echo "===== Generalization: kclique model trained on n5 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

echo "--- Generating missing kclique test scales (n3, n12) ---"
bash scripts/generate_gen_data.sh kclique

START=$(date +%s)
echo "--- Running generalization eval (n3 → n15) ---"
python scripts/eval_generalization.py \
    --family kclique \
    --train-scale n5 \
    --model-path experiments/models/kclique/n5/mlp_full_e_kl_sm/best_mlp_full_e_kl_sm.pt \
    --interian-model experiments/models/kclique/n5/interian/best_interian.pt \
    --model-label run7 \
    --split test \
    --test-scales n3 n5 n10 n12 n15 \
    --max-tries 10 \
    --max-instances 100 \
    --save-dir experiments/generalization/kclique_from_n5 \
    --methods minbreak noveltyplus interian ours_frozen ours_online_kl ours_online_success_kl \
    --verbose
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
