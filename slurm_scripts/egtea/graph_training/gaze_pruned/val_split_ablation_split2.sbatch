#!/bin/bash
#SBATCH --job-name=egtea_gpasg_split2_val_ablate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=40:00:00
#SBATCH --mem=40GB
#SBATCH --output=/home/s3758869/egocentric_video_graph_framework_ar/slurm_outputs/egtea_gpasg_split2_val_ablate_%A_%a.out
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
BASE_CONFIG="${REPO_ROOT}/graph_training/configs/egtea_gaze/action_recognition/per_split_gaze_pruned/split2.json"
GENERATED_DIR="${REPO_ROOT}/graph_training/configs/egtea_gaze/action_recognition/per_split_gaze_pruned_val_ablation/generated"

VARIANTS=(train10 test10 test1000)
VARIANT="${VARIANT_OVERRIDE:-${VARIANTS[$SLURM_ARRAY_TASK_ID]}}"
CONFIG_PATH="${GENERATED_DIR}/split2_${VARIANT}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

mkdir -p "${REPO_ROOT}/slurm_outputs" "${GENERATED_DIR}"

cd "${REPO_ROOT}"
source /home/s3758869/miniconda3/etc/profile.d/conda.sh
conda activate ego_graphs

/home/s3758869/miniconda3/envs/ego_graphs/bin/python - "${BASE_CONFIG}" "${CONFIG_PATH}" "${VARIANT}" <<'PY'
import json
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]

with base_path.open("r", encoding="utf-8") as f:
    cfg = json.load(f)

variant_settings = {
    "train10": {
        "description": "10% validation sampled stratified from train distribution",
        "val_fraction": 0.1,
        "val_distribution_source": None,
        "val_num_samples": None,
    },
    "test10": {
        "description": "10% validation sampled from train with action allocation resembling test distribution",
        "val_fraction": 0.1,
        "val_distribution_source": "test",
        "val_num_samples": None,
    },
    "test1000": {
        "description": "1000 validation clips sampled from train with action allocation resembling test distribution",
        "val_fraction": 0.1,
        "val_distribution_source": "test",
        "val_num_samples": 1000,
    },
}
if variant not in variant_settings:
    raise ValueError(f"Unknown variant: {variant}")

settings = variant_settings[variant]
data_cfg = cfg["data"]
data_cfg["split_ids"] = [2]
data_cfg["val_fraction"] = settings["val_fraction"]
if settings["val_distribution_source"] is None:
    data_cfg.pop("val_distribution_source", None)
else:
    data_cfg["val_distribution_source"] = settings["val_distribution_source"]
if settings["val_num_samples"] is None:
    data_cfg.pop("val_num_samples", None)
else:
    data_cfg["val_num_samples"] = settings["val_num_samples"]

cfg["experiment_name"] = f"egtea_gpasg_split2_val_{variant}_30ep"
cfg["training"]["final_checkpoint_metric"] = "mean_accuracy"
data_cfg["easg_cache_path"] = (
    "/home/s3758869/egocentric_video_graph_framework_ar/outputs/egtea_gaze/"
    f"action_recognition/easg_tensor_cache_gpasg_split2_val_ablate_{variant}"
)
cfg["output"]["base_path"] = (
    "/home/s3758869/egocentric_video_graph_framework_ar/outputs/egtea_gaze/"
    f"action_recognition/gpasg_split2_val_ablation/{variant}"
)
cfg["validation_ablation"] = {
    "variant": variant,
    "description": settings["description"],
    "split_id": 2,
    "graph_setup": "gaze-pruned action scene graph (GPASG)",
    "checkpoint_policy": "save best mean_accuracy, top5, and f1 on validation; evaluate all three on test",
    "expected_test_outputs": [
        "final_test_results_mean_accuracy.json",
        "final_test_results_top5.json",
        "final_test_results_f1.json",
    ],
    "source_config": str(base_path),
}

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"Wrote generated config: {out_path}", flush=True)
PY

echo "GPASG split2 validation ablation variant: ${VARIANT}"
echo "Config: ${CONFIG_PATH}"
nvidia-smi

python -u graph_training/train_graph_lstm_egtea.py \
    --config-path "${CONFIG_PATH}"
