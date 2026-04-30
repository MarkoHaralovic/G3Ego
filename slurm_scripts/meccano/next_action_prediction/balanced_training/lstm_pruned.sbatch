#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=balanced_lstm_pruned
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/balanced_training/lstm_pruned_%J.log

set -euo pipefail
mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/balanced_training
nvidia-smi

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
PYTHON_BIN="/home/s3758869/miniconda3/envs/ego_graphs/bin/python"
CONFIG_PATH="/home/s3758869/egocentric_video_graph_framework_ar/graph_training/configs/meccano/next_action_prediction/balanced_training/lstm_pruned.json"

cd "${REPO_ROOT}"
echo "Running balanced lstm_pruned"
"${PYTHON_BIN}" graph_training/train_graph_lstm_meccano.py --config-path "${CONFIG_PATH}"
