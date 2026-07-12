#!/bin/bash
#SBATCH --job-name=egtea_gaze_pruned_ohd_testdist_split1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=40:00:00
#SBATCH --mem=40GB
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/egtea_gaze_pruned_ohd_testdist_split1_%j.out
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
CONFIG_PATH="/home/s3758869/egocentric_video_graph_framework_ar/graph_training/configs/egtea_gaze/action_recognition/per_split_gaze_pruned_ohd_testdist/split1.json"

mkdir -p "${REPO_ROOT}/slurm_outputs"

cd "${REPO_ROOT}"
source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs

nvidia-smi

python -u graph_training/train_graph_lstm_egtea.py \
    --config-path "${CONFIG_PATH}"
