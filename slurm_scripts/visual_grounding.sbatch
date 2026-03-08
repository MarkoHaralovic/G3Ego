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

cd /home/s3758869/egocentric_video_graph_framework_ar/graph_construction/object_detection_pipeline/Open-GroundingDino

# running Grounding DINO Tiny
python extract_gdino_features.py  \
    -c tools/GroundingDINO_SwinT_OGC.py   \
    -p /home/s3758869/egocentric_video_graph_framework_ar/graph_construction/object_detection_pipeline/Open-GroundingDino/weights/gdinot-1.8m-odvg.pth   \
    -i /home/s3758869/vlm_datasets/AriaEA_vlm_ann_3_10_llava-v1.6-34b-hf   \
    --device cuda

# running Grounding DINO Base
python extract_gdino_features.py  \
    -c tools/GroundingDINO_SwinB_cfg.py   \
    -p /home/s3758869/egocentric_video_graph_framework_ar/graph_construction/object_detection_pipeline/Open-GroundingDino/weights/groundingdino_swinb_cogcoor.pth  \
    -i /home/s3758869/vlm_datasets/aaa \
    --device cuda