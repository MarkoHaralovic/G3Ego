#!/bin/bash
#SBATCH --job-name=egtea_visual_hand_grounding
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=240:00:00
#SBATCH --partition=main,students,main-gpu,dmb
#SBATCH --array=0-15
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/egtea_visual_hand_grounding_%A_%a.log

set -euo pipefail

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs

source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs
export PYTHONNOUSERSITE=1
unset PYTHONPATH
PYTHON_BIN="/home/s3758869/miniconda3/envs/ego_graphs/bin/python"

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
GDINO_DIR="/home/s3758869/egocentric_video_graph_framework_ar/graph_construction/object_detection_pipeline/Open-GroundingDino"
ANNOTATION_ROOT="/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/egtea_gaze/vlm_ann_Qwen3-VL-32B-Instruct"
FRAMES_ROOT="/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/egtea_gaze/framewise_videos"
HAND_MASK_ROOT="/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/egtea_gaze/hand_masks"
CONFIG_FILE="/home/s3758869/egocentric_video_graph_framework_ar/graph_construction/object_detection_pipeline/Open-GroundingDino/tools/GroundingDINO_SwinB_cfg.py"
CHECKPOINT="/home/s3758869/egocentric_video_graph_framework_ar/graph_construction/object_detection_pipeline/Open-GroundingDino/weights/groundingdino_swinb_cogcoor.pth"

nvidia-smi

cd "${GDINO_DIR}"

"${PYTHON_BIN}" -u extract_gdino_features_egtea.py \
    -c "${CONFIG_FILE}" \
    -p "${CHECKPOINT}" \
    --annotation-root "${ANNOTATION_ROOT}" \
    --frames-root "${FRAMES_ROOT}" \
    --hand-mask-root "${HAND_MASK_ROOT}" \
    --device cuda \
    --extract-hands \
    --overwrite

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -u graph_construction/object_detection_pipeline/get_gazed_at_objects_egtea.py \
    --input-root "${ANNOTATION_ROOT}" \
    --grounding-name grounding_results_gdino_base.pkl

echo "[$(date --iso-8601=seconds)] Finished EGTEA visual+hand grounding"
