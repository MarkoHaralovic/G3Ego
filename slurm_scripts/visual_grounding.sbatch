#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=visual_grounding
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/surgical_video_assessment/slurm_outputs/visual_grounding_%J.log

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

conda activate ego_graphs
cd graph_construction/object_detection_pipeline/Open-GroundingDino/models/GroundingDINO/ops
python setup.py build install
python test.py

cd /home/s3758869/egocentric_video_graph_framework_ar
python graph_training/train_graph_mlp.py 