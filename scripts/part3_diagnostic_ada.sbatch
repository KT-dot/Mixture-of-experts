#!/bin/bash
# Greedy vs beam-1 diagnostic on the same GPU type as the saved Part 3 runs.
#SBATCH --job-name=a2-p3diag
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
# The "phase3" 2080 Ti nodes failed CUDA initialisation (gnode066, gnode076, 2026-09-28).
#SBATCH --exclude=gnode[065-066,068,070,073,075-081,083,087-090]
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home2/%u/anlp_a2/a2_20260928/logs/part3-diag-%j.log
set -euo pipefail
source "$HOME/anlp_a2/slurm/env.sh"; cd "$HOME/anlp_a2/a2_20260928"
export TOKENIZERS_PARALLELISM=false
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
python scripts/part3_greedy_beam1_diagnostic.py --device cuda --output outputs/part3
