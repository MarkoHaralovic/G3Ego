#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch "${SCRIPT_DIR}/val_split_ablation_split2.sbatch"
