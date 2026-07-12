#!/bin/bash
set -euo pipefail

SCRIPT_DIR="/home/s3758869/egocentric_video_graph_framework_ar/slurm_scripts/meccano/next_action_prediction/optuna"

sbatch "${SCRIPT_DIR}/meccano_next_action_mlp_full_f1_optuna.sbatch"
sbatch "${SCRIPT_DIR}/meccano_next_action_mlp_pruned_f1_optuna.sbatch"
sbatch "${SCRIPT_DIR}/meccano_next_action_lstm_full_f1_optuna.sbatch"
sbatch "${SCRIPT_DIR}/meccano_next_action_lstm_pruned_f1_optuna.sbatch"
