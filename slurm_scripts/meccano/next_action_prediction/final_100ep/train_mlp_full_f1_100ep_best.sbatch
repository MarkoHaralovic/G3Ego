#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=na_mlp_full_f1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/na_mlp_full_f1_%J.log

set -euo pipefail

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs
cd /home/s3758869/egocentric_video_graph_framework_ar

source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs

python graph_training/train_graph_mlp_meccano_fixed.py \
  --config-path /home/s3758869/egocentric_video_graph_framework_ar/graph_training/configs/meccano/next_action_prediction/final_100ep/mlp_full_f1_100ep_best.json \
  --objective-metric f1
