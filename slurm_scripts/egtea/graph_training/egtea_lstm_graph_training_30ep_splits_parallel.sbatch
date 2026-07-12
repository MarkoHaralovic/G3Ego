#!/bin/bash
#SBATCH --job-name=egtea_lstm_30ep
#SBATCH --array=1-3%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=40:00:00
#SBATCH --mem=40GB
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/egtea_lstm_30ep_split%A_%a.out
#SBATCH --partition=main,students,main-gpu

set -euo pipefail

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/graph_training/configs/egtea_gaze/action_recognition/lstm_run_config.json}"
SPLIT_ID="${SLURM_ARRAY_TASK_ID}"
NUM_EPOCHS="${NUM_EPOCHS:-30}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-egtea_gaze_full_graph_lstm_best_30ep_parallel}"

mkdir -p "${REPO_ROOT}/slurm_outputs"

cd "${REPO_ROOT}"
source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs

echo "[$(date --iso-8601=seconds)] Starting EGTEA LSTM graph training"
echo "Host: $(hostname)"
echo "Array job: ${SLURM_ARRAY_JOB_ID:-<none>} task: ${SLURM_ARRAY_TASK_ID:-<none>}"
echo "Config: ${CONFIG_PATH}"
echo "Split id: ${SPLIT_ID}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Experiment: ${EXPERIMENT_NAME}"
nvidia-smi

python -u graph_training/train_graph_lstm_egtea.py \
  --config-path "${CONFIG_PATH}" \
  --split-ids "${SPLIT_ID}" \
  --num-epochs "${NUM_EPOCHS}" \
  --experiment-name "${EXPERIMENT_NAME}"

echo "[$(date --iso-8601=seconds)] Finished EGTEA LSTM graph training split ${SPLIT_ID}"
