#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch "${SCRIPT_DIR}/split1_testdist.sbatch"
sbatch "${SCRIPT_DIR}/split2_testdist.sbatch"
sbatch "${SCRIPT_DIR}/split3_testdist.sbatch"
