#!/bin/bash
# Re-score the five saved Part 1 checkpoints on the full test split (inference only).
#SBATCH --job-name=a2-p1bleu
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
# The "phase3" 2080 Ti nodes failed CUDA initialisation (gnode066, gnode076, 2026-09-28).
#SBATCH --exclude=gnode[065-066,068,070,073,075-081,083,087-090]
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=12:00:00
#SBATCH --output=/home2/%u/anlp_a2/a2_20260928/logs/part1-bleu-%j.log

set -euo pipefail
PART1="$HOME/anlp_a2/part1_20260922"
source "$HOME/anlp_a2/slurm/env.sh"
cd "$PART1"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
echo "=== PART 1 FULL-TEST BLEU START $(date --iso-8601=seconds) host=$(hostname)"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
papermill 10_full_test_bleu.ipynb "$HOME/anlp_a2/a2_20260928/logs/10_full_test_bleu.executed.ipynb" --cwd "$PART1" --log-output
cp "$HOME/anlp_a2/a2_20260928/logs/10_full_test_bleu.executed.ipynb" "$PART1/10_full_test_bleu.ipynb"
echo "=== PART 1 FULL-TEST BLEU DONE $(date --iso-8601=seconds)"
