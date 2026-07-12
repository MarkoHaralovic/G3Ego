#!/bin/bash
#SBATCH --job-name=egtea_gpasg_ohd_tempagg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=40:00:00
#SBATCH --mem=40GB
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/egtea_gpasg_ohd_tempagg_split%j.out
#SBATCH --partition=dmb,main,students,main-gpu

set -euo pipefail

SPLIT_ID="${1:?Usage: sbatch run_graph_tempagg_split.sbatch SPLIT_ID}"
case "${SPLIT_ID}" in
  1|2|3) ;;
  *)
    echo "Unsupported split id: ${SPLIT_ID}. Use 1, 2, or 3." >&2
    exit 2
    ;;
esac

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
CONFIG_PATH="${REPO_ROOT}/graph_training/configs/egtea_gaze/action_recognition/per_split_gaze_pruned_ohg_tempagg/split${SPLIT_ID}.json"

mkdir -p "${REPO_ROOT}/slurm_outputs"

cd "${REPO_ROOT}"
source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs

echo "Running EGTEA GPASG+OHD graph temporal aggregator split ${SPLIT_ID}"
echo "Config: ${CONFIG_PATH}"
nvidia-smi

python -u graph_training/train_graph_lstm_egtea.py \
    --config-path "${CONFIG_PATH}"
