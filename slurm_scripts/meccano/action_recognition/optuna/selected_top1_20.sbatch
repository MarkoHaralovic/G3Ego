#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=ar_top1_20
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano/action_recognition/optuna/selected_top1_20_%A_%a.log
#SBATCH --constraint="a40|a100|l40|l40s|h200nvl|rtx6000pro"

set -euo pipefail
mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano/action_recognition/optuna
nvidia-smi

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
PYTHON_BIN="/home/s3758869/miniconda3/envs/ego_graphs/bin/python"

MODEL_TYPES=("lstm" "lstm" "mlp" "mlp")
NAMES=("lstm_full" "lstm_pruned" "mlp_full" "mlp_pruned")
TRIALS=(2 26 13 1)

IDX="${SLURM_ARRAY_TASK_ID}"
MODEL_TYPE="${MODEL_TYPES[$IDX]}"
NAME="${NAMES[$IDX]}"
TRIAL="${TRIALS[$IDX]}"

BASE_CONFIG="${REPO_ROOT}/graph_training/configs/meccano/action_recognition/optuna_base/${NAME}.json"
OPTUNA_CONFIG="${REPO_ROOT}/graph_training/configs/meccano/optuna/action_recognition/${NAME}.json"
OUTPUT_BASE="${REPO_ROOT}/outputs/meccano/action_recognition/selected_top1_20/${NAME}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" graph_training/run_selected_meccano_optuna.py \
  --model-type "${MODEL_TYPE}" \
  --base-config-path "${BASE_CONFIG}" \
  --optuna-config-path "${OPTUNA_CONFIG}" \
  --trial-number "${TRIAL}" \
  --num-epochs 20 \
  --metric acc \
  --output-base-path "${OUTPUT_BASE}" \
  --eval-initial-checkpoint \
  --resume-optimizer
