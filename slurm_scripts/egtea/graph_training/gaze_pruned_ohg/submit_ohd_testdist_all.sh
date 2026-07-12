#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch "${SCRIPT_DIR}/split1_ohd_testdist.sbatch"
sbatch "${SCRIPT_DIR}/split2_ohd_testdist.sbatch"
sbatch "${SCRIPT_DIR}/split3_ohd_testdist.sbatch"
