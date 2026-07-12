#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=na_gpasg_lstm_rgb_ohg
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/na_gpasg_lstm_rgb_ohg_%J.log

set -euo pipefail

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs
cd /home/s3758869/egocentric_video_graph_framework_ar

DATA_ROOT="/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/MECCANO_vlm_ann_Qwen3-VL-32B-Instruct-3fps"
CACHE_ROOT="/home/s3758869/egocentric_video_graph_framework_ar/outputs/meccano/easg_tensor_cache_next_action_uniform_delta1s_obs3p5s_12fps_pruned_rgb_ohg"
OHG_ROOT="/home/s3758869/egocentric_video_graph_framework_ar/rulstm"

for path in \
  "${DATA_ROOT}/Train" \
  "${DATA_ROOT}/Val" \
  "${DATA_ROOT}/Test" \
  "${DATA_ROOT}/next_action_classification/MECCANO_train_next_actions_delta1s.csv" \
  "${DATA_ROOT}/next_action_classification/MECCANO_val_next_actions_delta1s.csv" \
  "${DATA_ROOT}/next_action_classification/MECCANO_test_next_actions_delta1s.csv" \
  "${OHG_ROOT}/RULSTM_MECCANO_trainval_features/obj" \
  "${OHG_ROOT}/RULSTM_MECCANO_trainval_features/gaze" \
  "${OHG_ROOT}/RULSTM_MECCANO_trainval_features/hands" \
  "${OHG_ROOT}/RULSTM_MECCANO_test_features/obj" \
  "${OHG_ROOT}/RULSTM_MECCANO_test_features/gaze" \
  "${OHG_ROOT}/RULSTM_MECCANO_test_features/hands"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${CACHE_ROOT}"

source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs
export PYTHONNOUSERSITE=1

python graph_training/train_graph_lstm_meccano.py \
  --config-path /home/s3758869/egocentric_video_graph_framework_ar/graph_training/configs/meccano/next_action_prediction/ohg_ablation/gpasg_next_action_lstm_rgb_ohg.json \
  --objective-metric top5
