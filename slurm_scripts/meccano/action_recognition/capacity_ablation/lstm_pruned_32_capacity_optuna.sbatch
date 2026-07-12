#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=ar_lstm_cap_hpo
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano/action_recognition/capacity_ablation/optuna_%J.log
#SBATCH --constraint="a40|a100|l40|l40s|h200nvl|rtx6000pro"

set -euo pipefail
mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano/action_recognition/capacity_ablation
mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/outputs/meccano/action_recognition/capacity_ablation/optuna
nvidia-smi

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
PYTHON_BIN="/home/s3758869/miniconda3/envs/ego_graphs/bin/python"
BASE_CONFIG="${REPO_ROOT}/graph_training/configs/meccano/action_recognition/capacity_ablation/lstm_pruned_32_adamw_optuna_base.json"
OPTUNA_CONFIG="${REPO_ROOT}/graph_training/configs/meccano/optuna/action_recognition/capacity_ablation/lstm_pruned_32_capacity_adamw.json"

cd "${REPO_ROOT}"
echo "Running action-recognition LSTM pruned 32 capacity/AdamW Optuna"
"${PYTHON_BIN}" graph_training/train_graph_lstm_meccano_optuna.py \
  --config-path "${BASE_CONFIG}" \
  --optuna-config-path "${OPTUNA_CONFIG}"
