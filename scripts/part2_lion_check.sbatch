#!/bin/bash
# Sensitivity check (not part of the selection protocol): full-pass Lion at the learning
# rates the short-budget sweep nearly tied with its pick. Usage: sbatch ... <lr>
#SBATCH --job-name=a2-lion
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH --exclude=gnode[065-066,068,070,073,075-081,083,087-090]
#SBATCH --cpus-per-task=6
#SBATCH --mem=30G
#SBATCH --time=04:00:00
#SBATCH --output=/home2/%u/anlp_a2/a2_20260928/logs/part2-lion-%j.log
set -euo pipefail
source "$HOME/anlp_a2/slurm/env.sh"; cd "$HOME/anlp_a2/a2_20260928"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
OUT=outputs/part2/v2_lion_check_lr$1
python main.py --part 2 --config configs/part2.yaml --optimizers lion --hyperparameters $OUT/lion_lr.json --output $OUT
