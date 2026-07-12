#!/bin/bash
#SBATCH --job-name=egtea_hand_grounding
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=240:00:00
#SBATCH --partition=main,students,main-gpu,dmb
#SBATCH --array=0-7
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/egtea_hand_grounding_%A_%a.log

set -euo pipefail

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p /home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs

source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs
export PYTHONNOUSERSITE=1
unset PYTHONPATH
PYTHON_BIN="/home/s3758869/miniconda3/envs/ego_graphs/bin/python"

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
GDINO_DIR="${REPO_ROOT}/graph_construction/object_detection_pipeline/Open-GroundingDino"
ANNOTATION_ROOT="${ANNOTATION_ROOT:-${REPO_ROOT}/data/egtea_gaze_plus/vlm_ann_Qwen3-VL-32B-Instruct}"
FRAMES_ROOT="${FRAMES_ROOT:-/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/egtea_gaze/framewise_videos}"
HAND_MASK_ROOT="${HAND_MASK_ROOT:-/projects/eemcs/dmb/ComputerVision/ego_graphs/vlm_datasets/egtea_gaze/hand_masks}"
CONFIG_FILE="${CONFIG_FILE:-${GDINO_DIR}/tools/GroundingDINO_SwinB_cfg.py}"
CHECKPOINT="${CHECKPOINT:-${GDINO_DIR}/weights/groundingdino_swinb_cogcoor.pth}"
HAND_EXTRA_ARGS="${HAND_EXTRA_ARGS:-}"

echo "[$(date --iso-8601=seconds)] Starting EGTEA hand grounding"
echo "Host: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-none}/${SLURM_ARRAY_TASK_COUNT:-none}"
echo "Annotation root: ${ANNOTATION_ROOT}"
echo "Frames root: ${FRAMES_ROOT}"
echo "Hand mask root: ${HAND_MASK_ROOT}"
nvidia-smi

cd "${GDINO_DIR}"

"${PYTHON_BIN}" -u extract_hand_features_egtea.py \
    -c "${CONFIG_FILE}" \
    -p "${CHECKPOINT}" \
    --annotation-root "${ANNOTATION_ROOT}" \
    --frames-root "${FRAMES_ROOT}" \
    --hand-mask-root "${HAND_MASK_ROOT}" \
    --device cuda \
    ${HAND_EXTRA_ARGS}

echo "[$(date --iso-8601=seconds)] Finished EGTEA hand grounding"
