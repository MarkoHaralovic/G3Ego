#!/bin/bash
#SBATCH --job-name=egtea_gaze_pruned_ohd
#SBATCH --array=1-3%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=40:00:00
#SBATCH --mem=40GB
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/egtea_gaze_pruned_ohd_%A_%a.out
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
SPLIT_ID="${SLURM_ARRAY_TASK_ID}"
CONFIG_PATH="${REPO_ROOT}/graph_training/configs/egtea_gaze/action_recognition/per_split_gaze_pruned_ohg/split${SPLIT_ID}.json"

mkdir -p "${REPO_ROOT}/slurm_outputs"

cd "${REPO_ROOT}"
source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs

echo "[$(date --iso-8601=seconds)] Starting EGTEA gaze-pruned OHD split ${SPLIT_ID}"
echo "Host: $(hostname)"
echo "Config: ${CONFIG_PATH}"
nvidia-smi

python -u graph_training/train_graph_lstm_egtea.py \
    --config-path "${CONFIG_PATH}"

echo "[$(date --iso-8601=seconds)] Finished EGTEA gaze-pruned OHD split ${SPLIT_ID}"
