#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=meccano_dinov3_feat_extraction
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano_dinov3_feat_extraction_%J.log

set -euo pipefail

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

cd /home/s3758869/egocentric_video_graph_framework_ar
source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs

cd /home/s3758869/egocentric_video_graph_framework_ar
python graph_construction/feature_extraction/dinov3_meccano.py
