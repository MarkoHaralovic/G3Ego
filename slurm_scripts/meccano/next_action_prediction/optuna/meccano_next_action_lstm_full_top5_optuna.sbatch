#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=na_lstm_full_top5_hpo
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano_next_action_lstm_full_top5_optuna_%J.log
#SBATCH --constraint="a40|a100|l40|l40s|h200nvl|rtx6000pro"

set -euo pipefail

nvidia-smi

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
PYTHON_BIN="/home/s3758869/miniconda3/envs/ego_graphs/bin/python"
BASE_CONFIG="${REPO_ROOT}/graph_training/configs/meccano/next_action_prediction/lstm_full_next_action.json"
OPTUNA_CONFIG="${REPO_ROOT}/graph_training/configs/meccano/optuna/next_action_prediction/lstm_full_top5.json"

mkdir -p "${REPO_ROOT}/slurm_outputs"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" graph_training/preflight_meccano_next_action.py \
  --config-path "${BASE_CONFIG}"

bash slurm_scripts/meccano/next_action_prediction/optuna/run_optuna_with_local_cache.sh \
  "${PYTHON_BIN}" \
  graph_training/train_graph_lstm_meccano_optuna.py \
  "${BASE_CONFIG}" \
  "${OPTUNA_CONFIG}"
