#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=ar_lstm_full_depth_concat_small
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/ar_lstm_full_depth_concat_small_%J.log
#SBATCH --constraint="a40|a100|l40|l40s|h200nvl|rtx6000pro"

set -euo pipefail

mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano/action_recognition/depth_ablation
mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/outputs/meccano/action_recognition/depth_ablation/dinov3_vits16_depth_dinov2_small/depth_concat
nvidia-smi

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

cd /home/s3758869/egocentric_video_graph_framework_ar

source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs
export PYTHONNOUSERSITE=1

python graph_training/train_graph_lstm_meccano.py \
  --config-path /home/s3758869/egocentric_video_graph_framework_ar/graph_training/configs/meccano/action_recognition/depth_concat/lstm_full_depth_concat.json \
  --objective-metric top5
