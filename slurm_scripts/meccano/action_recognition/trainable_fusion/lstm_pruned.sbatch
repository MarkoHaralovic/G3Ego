#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=ar_lstm_pruned_fuse
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano/action_recognition/trainable_fusion/lstm_pruned_%J.log
#SBATCH --constraint="a40|a100|l40|l40s|h200nvl|rtx6000pro"

set -euo pipefail
mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano/action_recognition/trainable_fusion
nvidia-smi

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
PYTHON_BIN="/home/s3758869/miniconda3/envs/ego_graphs/bin/python"
CONFIG_PATH="${REPO_ROOT}/graph_training/configs/meccano/action_recognition/trainable_fusion/lstm_pruned.json"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" graph_training/train_next_action_posthoc_meccano.py --config-path "${CONFIG_PATH}"
