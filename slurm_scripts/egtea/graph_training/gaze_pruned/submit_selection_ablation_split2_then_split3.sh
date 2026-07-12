#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/selection_ablation_split2_3.sbatch"

split2_job="$(
  sbatch --parsable \
    --array=0-2 \
    --export=ALL,SPLIT_ID=2 \
    "${SBATCH_SCRIPT}"
)"
echo "Submitted split2 selection ablation array: ${split2_job}"

split3_job="$(
  sbatch --parsable \
    --dependency=afterok:${split2_job} \
    --array=0-2 \
    --export=ALL,SPLIT_ID=3 \
    "${SBATCH_SCRIPT}"
)"
echo "Submitted split3 selection ablation array after split2: ${split3_job}"
