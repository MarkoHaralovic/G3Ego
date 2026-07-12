#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=graph_lstm_next_action_training
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano_lstm_next_action_training_%J.log
#SBATCH --constraint="a40|a100|l40|l40s|h200nvl"

set -euo pipefail

nvidia-smi

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
PYTHON_BIN="/home/s3758869/miniconda3/envs/ego_graphs/bin/python"
CONFIG_PATH="/home/s3758869/egocentric_video_graph_framework_ar/graph_training/configs/meccano/next_action_prediction/fsag_next_action_lstm.json"

cd "${REPO_ROOT}"

echo "Python: ${PYTHON_BIN}"
echo "Config: ${CONFIG_PATH}"

"${PYTHON_BIN}" graph_training/train_graph_lstm_meccano.py \
  --config-path "${CONFIG_PATH}"
