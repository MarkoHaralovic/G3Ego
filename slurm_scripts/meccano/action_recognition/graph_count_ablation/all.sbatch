#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=ar_graph_count
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano/action_recognition/graph_count_ablation/%A_%a.log
#SBATCH --constraint="a40|a100|l40|l40s|h200nvl|rtx6000pro"

set -euo pipefail
mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano/action_recognition/graph_count_ablation
mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/outputs/meccano/action_recognition/graph_count_ablation
nvidia-smi

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
PYTHON_BIN="/home/s3758869/miniconda3/envs/ego_graphs/bin/python"

CONFIG_NAMES=(
  "lstm_full" "lstm_full" 
  "lstm_pruned" "lstm_pruned" 
  "mlp_full" "mlp_full" 
  "mlp_pruned" "mlp_pruned" 
)
GRAPH_COUNTS=(16 32 16 32 16 32 16 32)

IDX="${SLURM_ARRAY_TASK_ID}"
CONFIG_NAME="${CONFIG_NAMES[$IDX]}"
GRAPH_COUNT="${GRAPH_COUNTS[$IDX]}"
CONFIG_PATH="${REPO_ROOT}/graph_training/configs/meccano/action_recognition/graph_count_ablation/${CONFIG_NAME}.json"

cd "${REPO_ROOT}"
echo "Running action-recognition graph-count ablation ${CONFIG_NAME} with ${GRAPH_COUNT} graph(s)"
"${PYTHON_BIN}" graph_training/run_meccano_graph_count_ablation.py \
  --config-path "${CONFIG_PATH}" \
  --graph-count "${GRAPH_COUNT}"
