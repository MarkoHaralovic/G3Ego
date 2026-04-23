#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=graph_mlp_next_action_optuna
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano_graph_mlp_next_action_optuna_%J.log
#SBATCH --constraint="a40|a100|l40|l40s|h200nvl|rtx6000pro"

set -euo pipefail

nvidia-smi

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
PYTHON_BIN="/home/s3758869/miniconda3/envs/ego_graphs/bin/python"
BASE_CONFIG="${REPO_ROOT}/graph_training/configs/meccano/fsag_next_action_optuna.json"
OPTUNA_CONFIG="${REPO_ROOT}/graph_training/configs/meccano/optuna/next_action_prediction/mlp_optuna.json"

cd "${REPO_ROOT}"

echo "Python: ${PYTHON_BIN}"
echo "Base config: ${BASE_CONFIG}"
echo "Optuna config: ${OPTUNA_CONFIG}"

"${PYTHON_BIN}" graph_training/train_graph_mlp_meccano_optuna.py \
  --config-path "${BASE_CONFIG}" \
  --optuna-config-path "${OPTUNA_CONFIG}"
