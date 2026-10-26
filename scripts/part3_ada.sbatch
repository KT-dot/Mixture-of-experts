#!/bin/bash
# Part 3 decoding evaluation on one ADA RTX 2080 Ti. Pass extra main.py arguments after
# the script name, e.g. `sbatch scripts/part3_ada.sbatch --strategies greedy beam_1`.
#SBATCH --job-name=a2-part3
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
# The "phase3" 2080 Ti nodes failed CUDA initialisation (gnode066, gnode076, 2026-09-28).
#SBATCH --exclude=gnode[065-066,068,070,073,075-081,083,087-090]
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home2/%u/anlp_a2/a2_20260928/logs/part3-%j.log

set -euo pipefail
PROJECT="$HOME/anlp_a2/a2_20260928"
source "$HOME/anlp_a2/slurm/env.sh"
cd "$PROJECT"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "=== PART 3 START $(date --iso-8601=seconds) host=$(hostname) job=$SLURM_JOB_ID args=$*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
python main.py --part 3 --config configs/part3.yaml "$@"
echo "=== PART 3 DONE $(date --iso-8601=seconds)"
