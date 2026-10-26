#!/bin/bash
# Run on the ADA login node once services are back. Needs a Hugging Face token with
# write access in $HF_HOME/token (already configured on ADA). Usage:
#   bash scripts/ada_finalize.sh <hf-namespace> [public|private]
set -euo pipefail
NAMESPACE="${1:?usage: ada_finalize.sh <hf-namespace> [public|private]}"
VISIBILITY="${2:-public}"
PROJECT="$HOME/anlp_a2/a2_20260928"
PART1="$HOME/anlp_a2/part1_20260922"
cd "$PROJECT"

echo "=== Part 2 checkpoint hashes"
sha256sum outputs/part2/v2/checkpoints/*/fraction_1.0.pt | tee outputs/part2/v2/checkpoint_sha256.txt

echo "=== Part 3 greedy vs beam-1 diagnostic (GPU job)"
sbatch scripts/part3_diagnostic_ada.sbatch

echo "=== Hugging Face publication ($VISIBILITY) from a compute node"
# Part 1 was already published from the laptop on 2026-10-01; set UPLOAD_PART1=1 to redo it.
srun -A research -p u22 -q medium -c 2 --mem=8G -t 01:00:00 bash -lc "
  source \$HOME/anlp_a2/slurm/env.sh && cd $PROJECT &&
  if [ '${UPLOAD_PART1:-0}' = 1 ]; then
    python scripts/upload_to_hf.py --namespace $NAMESPACE --part 1 --part1-root $PART1 \
      --visibility $VISIBILITY --execute | tee logs/hf_part1.json
  fi &&
  python scripts/upload_to_hf.py --namespace $NAMESPACE --part 2 --run outputs/part2/v2 \
    --visibility $VISIBILITY --execute | tee logs/hf_part2.json"
