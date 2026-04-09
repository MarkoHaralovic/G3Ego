#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=meccano_visual_grounding
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano_visual_grounding_%J.log

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

set -euo pipefail

source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs
export PYTHONNOUSERSITE=1
unset PYTHONPATH
PYTHON_BIN="/home/s3758869/miniconda3/envs/ego_graphs/bin/python"

cd /home/s3758869/egocentric_video_graph_framework_ar/graph_construction/object_detection_pipeline/Open-GroundingDino

# running Grounding DINO Base
"${PYTHON_BIN}" -s extract_gdino_features_meccano.py \
    -c tools/GroundingDINO_SwinB_cfg.py \
    -p /home/s3758869/egocentric_video_graph_framework_ar/graph_construction/object_detection_pipeline/Open-GroundingDino/weights/groundingdino_swinb_cogcoor.pth \
    --device cuda
