#!/bin/bash
#SBATCH --job-name=egtea_gpasg_split1_testdist_hpo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --array=0-3%2
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/egtea_gpasg_split1_testdist_hpo_%A_%a.out
#SBATCH --partition=main,students,main-gpu,dmb

set -euo pipefail

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
BASE_CONFIG="/home/s3758869/egocentric_video_graph_framework_ar/graph_training/configs/egtea_gaze/action_recognition/per_split_gaze_pruned_testdist/split1.json"
OPTUNA_CONFIG="/home/s3758869/egocentric_video_graph_framework_ar/graph_training/configs/egtea_gaze/action_recognition/optuna/gaze_pruned_split1_testdist_lstm.json"
OUTPUT_ROOT="/home/s3758869/egocentric_video_graph_framework_ar/outputs/egtea_gaze/action_recognition/lstm_gaze_pruned_split1_testdist_hpo"

mkdir -p "${REPO_ROOT}/slurm_outputs"

cd "${REPO_ROOT}"
source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs
export PYTHONNOUSERSITE=1

nvidia-smi

python -u graph_training/train_graph_lstm_egtea_optuna.py \
    --config-path "${BASE_CONFIG}" \
    --optuna-config-path "${OPTUNA_CONFIG}" \
    --split-ids 1 \
    --trials-to-run 6 \
    --output-base-path "${OUTPUT_ROOT}"
