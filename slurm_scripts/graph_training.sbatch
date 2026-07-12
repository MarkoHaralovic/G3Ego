#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=graph_training
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/graph_training_%J.log

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

conda activate ego_graphs

cd /home/s3758869/egocentric_video_graph_framework_ar
python graph_training/train_graph_mlp.py \
  --config-path /home/s3758869/egocentric_video_graph_framework_ar/graph_training/configs/fasg_run_config.json