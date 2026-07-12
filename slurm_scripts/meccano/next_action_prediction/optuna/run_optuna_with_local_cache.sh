#!/bin/bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <python_bin> <trainer> <base_config> <optuna_config>" >&2
  exit 2
fi

PYTHON_BIN="$1"
TRAINER="$2"
BASE_CONFIG="$3"
OPTUNA_CONFIG="$4"

JOB_ID="${SLURM_JOB_ID:-manual_$$}"
JOB_TMP="${TMPDIR:-/tmp}/${USER}/meccano_next_action_optuna_${JOB_ID}"
mkdir -p "${JOB_TMP}"

RUNTIME_CONFIG="${JOB_TMP}/$(basename "${BASE_CONFIG}")"
LOCAL_EASG_CACHE="${JOB_TMP}/easg_cache"

echo "Runtime config: ${RUNTIME_CONFIG}"
echo "Local EASG cache: ${LOCAL_EASG_CACHE}"

"${PYTHON_BIN}" - "${BASE_CONFIG}" "${RUNTIME_CONFIG}" "${LOCAL_EASG_CACHE}" <<'PY'
import json
import os
import sys

base_config, runtime_config, local_cache = sys.argv[1:4]
with open(base_config, "r") as f:
    cfg = json.load(f)

cfg.setdefault("data", {})
cfg["data"]["easg_cache_path"] = local_cache
cfg["data"]["num_workers"] = 0
cfg["data"]["pin_memory"] = False

os.makedirs(os.path.dirname(runtime_config), exist_ok=True)
with open(runtime_config, "w") as f:
    json.dump(cfg, f, indent=2)
PY

"${PYTHON_BIN}" "${TRAINER}" \
  --config-path "${RUNTIME_CONFIG}" \
  --optuna-config-path "${OPTUNA_CONFIG}"
