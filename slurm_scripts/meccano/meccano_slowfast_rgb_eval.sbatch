#!/bin/bash
#SBATCH --partition=main,students
#SBATCH --job-name=meccano_slowfast_rgb_eval
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|a100|l40|l40s|h200nvl"
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/meccano_slowfast_rgb_eval_%J.log

set -euo pipefail

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

ROOT=/home/s3758869/egocentric_video_graph_framework_ar
SLOWFAST_ROOT="${ROOT}/SlowFast"
DATA_ROOT="${ROOT}/MECCANO/dataset"
OUT_DIR="${SLOWFAST_ROOT}/outputs/meccano_slowfast_rgb_eval"
PRED_FILE="rgb_test_predictions.pkl"

mkdir -p "${ROOT}/slurm_outputs" "${OUT_DIR}"

source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs

cd "${SLOWFAST_ROOT}"
export PYTHONPATH="${SLOWFAST_ROOT}:${PYTHONPATH:-}"

nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python tools/run_net.py \
  --cfg configs/MECCANO/configs/action_recognition/SLOWFAST_8x8_R50_MECCANO.yaml \
  --opts \
  TRAIN.ENABLE False \
  TEST.ENABLE True \
  TEST.BATCH_SIZE 32 \
  TEST.CHECKPOINT_FILE_PATH /home/s3758869/egocentric_video_graph_framework_ar/SlowFast/checkpoints/SLOWFAST_8x8_R50_RGB_MECCANO.pyth \
  TEST.CHECKPOINT_TYPE pytorch \
  TEST.NUM_ENSEMBLE_VIEWS 10 \
  TEST.NUM_SPATIAL_CROPS 3 \
  DATA.PATH_TO_DATA_DIR "${DATA_ROOT}" \
  OUTPUT_DIR "${OUT_DIR}" \
  TEST.SAVE_RESULTS_PATH "${PRED_FILE}" \
  DATA_LOADER.NUM_WORKERS 4 \
  DATA_LOADER.PIN_MEMORY False \
  TENSORBOARD.ENABLE False

python - <<'PY'
import os
import pickle

import torch
from sklearn.metrics import f1_score

pred_path = "/home/s3758869/egocentric_video_graph_framework_ar/SlowFast/outputs/meccano_slowfast_rgb_eval/rgb_test_predictions.pkl"
print("prediction_file:", pred_path)
print("prediction_file_exists:", os.path.exists(pred_path))

preds, labels = pickle.load(open(pred_path, "rb"))
if not torch.is_tensor(preds):
    preds = torch.as_tensor(preds)
if not torch.is_tensor(labels):
    labels = torch.as_tensor(labels)

if labels.ndim > 1:
    labels = labels.argmax(dim=1)
labels = labels.long()

y_pred = preds.argmax(dim=1)
top1 = (y_pred == labels).float().mean().item() * 100.0
top5 = (preds.topk(5, dim=1).indices == labels[:, None]).any(dim=1).float().mean().item() * 100.0
macro_f1 = f1_score(labels.cpu().numpy(), y_pred.cpu().numpy(), average="macro", zero_division=0) * 100.0

print(f"MECCANO RGB SlowFast test top1={top1:.4f}")
print(f"MECCANO RGB SlowFast test top5={top5:.4f}")
print(f"MECCANO RGB SlowFast test macro_f1={macro_f1:.4f}")
PY
