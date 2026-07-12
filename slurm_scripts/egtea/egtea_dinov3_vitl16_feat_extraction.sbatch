#!/bin/bash
#SBATCH --job-name=egtea_dinov3_vitl16
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=main,students,main-gpu,dmb
#SBATCH --array=0-7
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/egtea_dinov3_vitl16_%A_%a.log

set -euo pipefail


# Configure HTTP proxy necessary for HPC internet connection
export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=http://proxy.utwente.nl:3128
export http_proxy=http://proxy.utwente.nl:3128
export https_proxy=http://proxy.utwente.nl:3128

export HF_HOME=/home/s3758869/.cache/huggingface
export HF_TOKEN=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

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

export HF_HOME="/projects/eemcs/dmb/ComputerVision/ego_graphs/model_cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
mkdir -p "${TRANSFORMERS_CACHE}"

cd /home/s3758869/egocentric_video_graph_framework_ar
source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs

echo "[$(date --iso-8601=seconds)] Starting EGTEA DINOv3 ViT-L/16 extraction"
echo "Host: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-none}/${SLURM_ARRAY_TASK_COUNT:-none}"
nvidia-smi

python -u graph_construction/feature_extraction/dinov3_egtea.py \
  --model-id facebook/dinov3-vitl16-pretrain-lvd1689m \
  --model-name dinov3_vitl16 \
  --cache-dir "${TRANSFORMERS_CACHE}" \
  --batch-size 192 \
  --num-workers "${SLURM_CPUS_PER_TASK:-16}" \
  --prefetch-factor 4 \
  --image-size 224 \
  --pooling mean \
  --no-amp \
  --overwrite \
  --save-dtype float32

echo "[$(date --iso-8601=seconds)] Finished EGTEA DINOv3 ViT-L/16 extraction"
