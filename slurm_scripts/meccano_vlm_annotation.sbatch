#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=meccano_qwen_vlm_ann
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano_qwen_vlm_ann_%J.log

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

conda activate ego_graphs

cd /home/s3758869/egocentric_video_graph_framework_ar/vlm_annotation

# Qwen/Qwen3-VL-32B-Instruct
python vlm_annotation_pipeline.py \
  --config qwen35vl/configs/qwen3_vl_32b_instruct.json