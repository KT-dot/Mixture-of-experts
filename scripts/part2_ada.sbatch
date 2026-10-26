#!/bin/bash
# Part 2 on one ADA RTX 2080 Ti: learning-rate sweep (bonus), then one full pass per
# optimizer at the selected learning rates. Submit with: sbatch scripts/part2_ada.sbatch
#SBATCH --job-name=a2-part2
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
# The "phase3" 2080 Ti nodes failed CUDA initialisation (gnode066, gnode076, 2026-09-28).
#SBATCH --exclude=gnode[065-066,068,070,073,075-081,083,087-090]
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=24:00:00
#SBATCH --output=/home2/%u/anlp_a2/a2_20260928/logs/part2-%j.log

set -euo pipefail
PROJECT="$HOME/anlp_a2/a2_20260928"
OUTPUT="${PART2_OUTPUT:-outputs/part2/v2}"
source "$HOME/anlp_a2/slurm/env.sh"
cd "$PROJECT"
mkdir -p outputs/part2/cache/huggingface
export TOKENIZERS_PARALLELISM=false MPLBACKEND=Agg OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "=== PART 2 START $(date --iso-8601=seconds) host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"

echo "=== SYNTHETIC CHECK"
python main.py --part 2 --synthetic-smoke --output outputs/part2/synthetic_smoke

if [ ! -f "$OUTPUT/sweep/selected_hyperparameters.json" ]; then
  echo "=== LEARNING-RATE SWEEP $(date --iso-8601=seconds)"
  python main.py --part 2 --config configs/part2.yaml --sweep --output "$OUTPUT"
fi
cat "$OUTPUT/sweep/selected_hyperparameters.json"

echo "=== FULL ONE-PASS RUNS $(date --iso-8601=seconds)"
python main.py --part 2 --config configs/part2.yaml \
  --hyperparameters "$OUTPUT/sweep/selected_hyperparameters.json" --output "$OUTPUT"
find "$OUTPUT" -maxdepth 3 -type f -printf '%p %s bytes\n' | sort
echo "=== PART 2 DONE $(date --iso-8601=seconds)"
