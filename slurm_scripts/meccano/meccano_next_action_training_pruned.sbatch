#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=graph_next_action_training_pruned
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano_next_action_training_pruned_%J.log
#SBATCH --constraint="a40|a100|l40|l40s|h200nvl"

nvidia-smi

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

cd /home/s3758869/egocentric_video_graph_framework_ar
python graph_training/train_graph_mlp_meccano.py --config graph_training/configs/meccano/next_action_prediction/gpasg_next_action.json