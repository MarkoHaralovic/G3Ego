#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=graph_training
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=8
#SBATCH --time=240:00:00
#SBATCH --gres=gpu:1
#SBATCH --output /home/s3758869/surgical_video_assessment/slurm_outputs/graph_training_%J.log

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

source /home/s3758869/video_surgical/bin/activate

cd /home/s3758869/surgical_video_assessment
python scripts/tapis_inference.py \
  --input-dir /home/s3758869/surgical_video_assessment/turbo_data/VideoAI \
  --video-name e1308a6d-2a86-4f29-8dd7-c5259bd04539_1556272 \
  --output-dir /home/s3758869/surgical_video_assessment/daVinci_tool_detection/tool_predictions/grasp/e1308a6d-2a86-4f29-8dd7-c5259bd04539_1556272