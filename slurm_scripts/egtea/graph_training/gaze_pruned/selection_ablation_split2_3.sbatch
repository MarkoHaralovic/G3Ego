#!/bin/bash
#SBATCH --job-name=egtea_gaze_pruned_sel_ablate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=40:00:00
#SBATCH --mem=40GB
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/egtea_gaze_pruned_sel_ablate_%A_%a.out
#SBATCH --partition=main,students,main-gpu
#SBATCH --array=0-2

set -euo pipefail

export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REPO_ROOT="/home/s3758869/egocentric_video_graph_framework_ar"
SPLIT_ID="${SPLIT_ID:-2}"
METRICS=(mean_accuracy top5 f1)
METRIC="${METRIC_OVERRIDE:-${METRICS[$SLURM_ARRAY_TASK_ID]}}"
BASE_CONFIG="${REPO_ROOT}/graph_training/configs/egtea_gaze/action_recognition/per_split_gaze_pruned/split${SPLIT_ID}.json"
GENERATED_DIR="${REPO_ROOT}/graph_training/configs/egtea_gaze/action_recognition/per_split_gaze_pruned_selection_ablation/generated"
CONFIG_PATH="${GENERATED_DIR}/split${SPLIT_ID}_${METRIC}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

mkdir -p "${REPO_ROOT}/slurm_outputs" "${GENERATED_DIR}"

cd "${REPO_ROOT}"
source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs

/home/s3758869/miniconda3/envs/ego_graphs/bin/python - "${BASE_CONFIG}" "${CONFIG_PATH}" "${SPLIT_ID}" "${METRIC}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
split_id = int(sys.argv[3])
metric = sys.argv[4]

with base_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

metric_tag = metric.replace("_", "")
cfg["experiment_name"] = f"egtea_gaze_pruned_lstm_split{split_id}_select_{metric_tag}_30ep"
cfg["data"]["split_ids"] = [split_id]
cfg["data"]["easg_cache_path"] = (
    "/home/s3758869/egocentric_video_graph_framework_ar/outputs/egtea_gaze/"
    f"action_recognition/easg_tensor_cache_gaze_pruned_sel_ablate_split{split_id}_{metric_tag}"
)
cfg["training"]["final_checkpoint_metric"] = metric
cfg["output"]["base_path"] = (
    "/home/s3758869/egocentric_video_graph_framework_ar/outputs/egtea_gaze/"
    f"action_recognition/lstm_gaze_pruned_selection_ablation/{metric_tag}"
)
cfg["selection_ablation"] = {
    "split_id": split_id,
    "final_checkpoint_metric": metric,
    "evaluates_checkpoints": ["mean_accuracy", "top5", "f1"],
    "source_config": str(base_path),
}

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"Wrote generated config: {out_path}", flush=True)
PY

echo "Split: ${SPLIT_ID}"
echo "Selection metric: ${METRIC}"
echo "Config: ${CONFIG_PATH}"
nvidia-smi

python -u graph_training/train_graph_lstm_egtea.py \
    --config-path "${CONFIG_PATH}"
