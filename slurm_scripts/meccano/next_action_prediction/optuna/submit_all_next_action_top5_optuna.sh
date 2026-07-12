#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch "${SCRIPT_DIR}/meccano_next_action_mlp_full_top5_optuna.sbatch"
sbatch "${SCRIPT_DIR}/meccano_next_action_mlp_pruned_top5_optuna.sbatch"
sbatch "${SCRIPT_DIR}/meccano_next_action_lstm_full_top5_optuna.sbatch"
sbatch "${SCRIPT_DIR}/meccano_next_action_lstm_pruned_top5_optuna.sbatch"
