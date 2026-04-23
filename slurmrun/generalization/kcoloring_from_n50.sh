#!/bin/bash
#SBATCH --job-name=GEN_KC_50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -A jeisner1
#SBATCH --mem-per-cpu=18GB
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/gen_kcoloring_from_n50_%j.out
#SBATCH --error=logs/slurm/gen_kcoloring_from_n50_%j.err

source /home/psingh54/.bashrc
module load anaconda3/2024.02-1
conda activate prabhav2
cd /home/psingh54/scratchjeisner1/psingh54/LearningToFlip
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
set -e

mkdir -p logs/slurm
mkdir -p experiments/generalization/kcoloring_from_n50

echo "===== Generalization: kcoloring model trained on n50 ====="
echo "Start: $(date)"
TOTAL_START=$(date +%s)

echo "--- Generating missing kcoloring test scales (n10, n20, n30) ---"
bash scripts/generate_gen_data.sh kcoloring

START=$(date +%s)
echo "--- Running generalization eval (n10 → n100) ---"
python scripts/eval_generalization.py \
    --family kcoloring \
    --train-scale n50 \
    --model-path experiments/models/kcoloring/n50/mlp_full_e_kl_sm/best_mlp_full_e_kl_sm.pt \
    --interian-model experiments/models/kcoloring/n50/interian/best_interian.pt \
    --model-label run7 \
    --split test \
    --test-scales n10 n20 n30 n50 n75 n100 \
    --max-tries 10 \
    --max-instances 100 \
    --save-dir experiments/generalization/kcoloring_from_n50 \
    --methods minbreak noveltyplus interian ours_frozen ours_online_kl ours_online_success_kl \
    --verbose
END=$(date +%s)
echo "  Eval time: $((END - START))s"

TOTAL_END=$(date +%s)
echo "===== Done. Total: $((TOTAL_END - TOTAL_START))s ====="
echo "End: $(date)"
