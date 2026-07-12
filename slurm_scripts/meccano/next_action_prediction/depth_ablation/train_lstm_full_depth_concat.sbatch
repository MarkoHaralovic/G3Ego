#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=na_lstm_full_depth_concat_small
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/na_lstm_full_depth_concat_small_%J.log

set -euo pipefail

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs
mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/outputs/meccano/next_action_prediction/depth_ablation/dinov3_vits16_depth_dinov2_small/depth_concat
cd /home/s3758869/egocentric_video_graph_framework_ar

source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs
export PYTHONNOUSERSITE=1

python graph_training/train_graph_lstm_meccano.py \
  --config-path /home/s3758869/egocentric_video_graph_framework_ar/graph_training/configs/meccano/next_action_prediction/depth_concat/lstm_full_next_action.json \
  --objective-metric top5
