#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=ar_lstm_cap
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-10
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano/action_recognition/capacity_ablation/%A_%a.log
#SBATCH --constraint="a40|a100|l40|l40s|h200nvl|rtx6000pro"

set -euo pipefail
mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano/action_recognition/capacity_ablation
mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/outputs/meccano/action_recognition/capacity_ablation
nvidia-smi

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
PYTHON_BIN="/home/s3758869/miniconda3/envs/ego_graphs/bin/python"
CONFIG_PATH="${REPO_ROOT}/graph_training/configs/meccano/action_recognition/capacity_ablation/lstm_pruned_32_capacity_matrix.json"

VARIANTS=(
  "baseline"
  "e1_emb64_widths32_k2"
  "e2_emb32_widths64_k2"
  "e3_emb64_widths64_k2"
  "e4_emb64_widths64_k4"
  "e5_e4_final256"
  "l1_e4_lstm_hidden192"
  "l2_e4_lstm_hidden256"
  "l3_e4_bidir_hidden128"
  "l3_e4_bidir_hidden192"
  "l4_e4_attention_readout"
)

IDX="${SLURM_ARRAY_TASK_ID}"
VARIANT="${VARIANTS[$IDX]}"

cd "${REPO_ROOT}"
echo "Running action-recognition LSTM capacity ablation variant ${VARIANT}"
"${PYTHON_BIN}" graph_training/run_meccano_capacity_ablation.py \
  --config-path "${CONFIG_PATH}" \
  --variant "${VARIANT}" \
  --run-test \
  --objective-metric top5
