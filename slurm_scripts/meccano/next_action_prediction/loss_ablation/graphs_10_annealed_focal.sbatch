#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=na_loss_g10
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano/next_action_recognition/loss_ablation/graphs_10_annealed_focal_%A_%a.log
#SBATCH --constraint="a40|a100|l40|l40s|h200nvl|rtx6000pro"

set -euo pipefail
mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano/next_action_recognition/loss_ablation
mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/outputs/meccano/next_action_recognition/loss_ablation
nvidia-smi

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
PYTHON_BIN="/home/s3758869/miniconda3/envs/ego_graphs/bin/python"

CONFIG_NAMES=(
  "lstm_full"
  "lstm_pruned"
  "mlp_full"
  "mlp_pruned"
)

IDX="${SLURM_ARRAY_TASK_ID}"
CONFIG_NAME="${CONFIG_NAMES[$IDX]}"
CONFIG_PATH="${REPO_ROOT}/graph_training/configs/meccano/next_action_prediction/loss_ablation/${CONFIG_NAME}.json"

cd "${REPO_ROOT}"
echo "Running next-action 10-graph annealed focal loss ablation ${CONFIG_NAME}"
"${PYTHON_BIN}" graph_training/run_meccano_next_action_loss_ablation.py \
  --config-path "${CONFIG_PATH}" \
  --graph-count 10 \
  --loss-name annealed_focal \
  --num-epochs 20
