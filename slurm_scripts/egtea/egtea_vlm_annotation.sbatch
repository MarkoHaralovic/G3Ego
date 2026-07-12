#!/bin/bash
#SBATCH --job-name=egtea_vlm_ann
#SBATCH --nodes=1
#SBATCH --gres=gpu:blackwell:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=100:00:00
#SBATCH --partition=dmb,main-gpu
#SBATCH --array=0-7
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/egtea_vlm_annotation_%A_%a.log

set -euo pipefail

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME="/projects/eemcs/dmb/ComputerVision/ego_graphs/model_cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/transformers"
mkdir -p "${HF_HUB_CACHE}"
mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs

cd /home/s3758869/egocentric_video_graph_framework_ar
source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs

CONFIG="${CONFIG:-/home/s3758869/egocentric_video_graph_framework_ar/vlm_annotation/qwen35vl/configs/qwen3_vl_32b_instruct_egtea.json}"

echo "[$(date --iso-8601=seconds)] Starting EGTEA VLM annotation"
echo "Host: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-none}/${SLURM_ARRAY_TASK_COUNT:-none}"
echo "Config: ${CONFIG}"
nvidia-smi

/home/s3758869/miniconda3/envs/ego_graphs/bin/python -u \
  vlm_annotation/egtea_vlm_annotation_pipeline.py \
  --config "${CONFIG}"

echo "[$(date --iso-8601=seconds)] Finished EGTEA VLM annotation"
