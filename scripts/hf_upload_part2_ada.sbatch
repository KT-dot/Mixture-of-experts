#!/bin/bash
# Publish the five Part 2 checkpoints to the Hugging Face Hub from a compute node
# (the gateway has no venv). Token: $HF_HOME/token, set by slurm/env.sh.
#SBATCH --job-name=a2-hfup2
#SBATCH --account=research
#SBATCH --partition=u22
#SBATCH --qos=medium
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/home2/%u/anlp_a2/a2_20260928/logs/hf-upload-part2-%j.log
set -euo pipefail
source "$HOME/anlp_a2/slurm/env.sh"; cd "$HOME/anlp_a2/a2_20260928"
NAMESPACE="${1:-Devatri}"
VISIBILITY="${2:-public}"
python scripts/upload_to_hf.py --namespace "$NAMESPACE" --part 2 --run outputs/part2/v2 \
  --visibility "$VISIBILITY" --execute | tee logs/hf_part2.json
